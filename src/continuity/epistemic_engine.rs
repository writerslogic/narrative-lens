use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    sync::OnceLock,
};

use serde::{Deserialize, Serialize};

use super::author_knowledge::{
    self, AuthorAssertion, AuthorAudience, AuthorKnowledgeAction, AuthorKnowledgeClassification,
    AuthorKnowledgeEvent, AuthorKnowledgeLedger, AuthorProposition, AuthorUptake, TruthValue,
};
use crate::{
    Error, Result,
    craft::lexicon::WordMatcher,
    manuscript::{
        Manuscript, TextRange,
        config::{validate_id, validate_name},
    },
    substrate::utils::{AnalysisAvailability, sentence_spans},
};

pub const MAX_EPISTEMIC_SCENES: usize = 10_000;
pub const MAX_EPISTEMIC_CHARACTERS: usize = 256;
pub const MAX_EPISTEMIC_SENTENCES: usize = 10_000;
pub const MAX_EPISTEMIC_MARKERS: usize = 100_000;
pub const MAX_EPISTEMIC_TEXT_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_EPISTEMIC_TOTAL_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_AUTHOR_EPISTEMIC_QUERIES: usize = 10_000;
pub const MAX_AUTHOR_EPISTEMIC_QUERY_BYTES: usize = 4 * 1024 * 1024;

fn serialize_sorted_ids<S: serde::Serializer>(
    ids: &HashSet<usize>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    let mut ids: Vec<_> = ids.iter().copied().collect();
    ids.sort_unstable();
    ids.serialize(serializer)
}

fn serialize_sorted_states<S: serde::Serializer>(
    states: &HashMap<String, EpistemicState>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    states
        .iter()
        .collect::<BTreeMap<_, _>>()
        .serialize(serializer)
}

/// A legacy lexical sentence candidate, not a verified story-world fact.
/// Importance is an uncalibrated marker tier, not probability or information bits.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
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
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EpistemicState {
    pub character: String,
    #[serde(serialize_with = "serialize_sorted_ids")]
    pub knows: HashSet<usize>,
    #[serde(serialize_with = "serialize_sorted_ids")]
    pub believes_false: HashSet<usize>,
    #[serde(serialize_with = "serialize_sorted_ids")]
    pub uncertain_about: HashSet<usize>,
}

/// Legacy reader state shape; prose alone cannot establish knowledge.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReaderEpistemicState {
    #[serde(serialize_with = "serialize_sorted_ids")]
    pub knows: HashSet<usize>,
    pub scene: usize,
}

/// Epistemic asymmetry measurement.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
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
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
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
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SuspensePoint {
    pub scene: usize,
    pub suspense_level: f64,
    pub source: String,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EpistemicResult {
    #[serde(default)]
    pub candidates: Vec<EpistemicCandidate>,
    #[serde(default)]
    pub analyzed_sentences: usize,
    #[serde(default)]
    pub candidate_scene_coverage: Option<f64>,
    #[serde(default)]
    pub semantic_metrics_available: bool,
    #[serde(default)]
    pub availability: crate::substrate::utils::AnalysisAvailability,
    #[serde(default)]
    pub diagnostics: Vec<String>,

    pub facts: Vec<StoryFact>,
    #[serde(serialize_with = "serialize_sorted_states")]
    pub character_states: HashMap<String, EpistemicState>,
    pub asymmetries: Vec<EpistemicAsymmetry>,
    pub revelations: Vec<RevelationEvent>,
    pub suspense_curve: Vec<SuspensePoint>,
    pub peak_suspense_scene: usize,
    pub information_density: f64,
    pub manipulation_score: f64,
    pub epistemic_insights: Vec<String>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicSignal {
    NamedCharacter,
    Secrecy,
    Plan,
    Identity,
    Relationship,
    Event,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EpistemicCandidate {
    pub fact_id: usize,
    pub scene: usize,
    pub range: TextRange,
    pub signals: Vec<EpistemicSignal>,
    pub mentioned_characters: Vec<String>,
}

fn signal_matchers() -> &'static [(EpistemicSignal, WordMatcher)] {
    static MATCHERS: OnceLock<Vec<(EpistemicSignal, WordMatcher)>> = OnceLock::new();
    MATCHERS.get_or_init(|| {
        use EpistemicSignal::*;
        [
            (
                Secrecy,
                vec![
                    "secret",
                    "hidden",
                    "concealed",
                    "nobody knew",
                    "no one knew",
                    "unknown",
                    "private",
                    "confidential",
                ],
            ),
            (
                Plan,
                vec![
                    "planned",
                    "intended",
                    "would soon",
                    "scheme",
                    "plot",
                    "strategy",
                    "meant to",
                ],
            ),
            (
                Identity,
                vec![
                    "actually was",
                    "real name",
                    "true identity",
                    "disguised",
                    "pretending",
                    "in truth",
                    "really",
                ],
            ),
            (
                Relationship,
                vec![
                    "loved", "hated", "betrayed", "trusted", "married", "siblings", "father",
                    "mother", "enemy", "ally",
                ],
            ),
            (
                Event,
                vec![
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
                ],
            ),
        ]
        .into_iter()
        .map(|(signal, words)| (signal, WordMatcher::new(&words)))
        .collect()
    })
}

fn empty_lexical_result(
    availability: AnalysisAvailability,
    diagnostics: Vec<String>,
) -> EpistemicResult {
    EpistemicResult {
        candidates: Vec::new(),
        analyzed_sentences: 0,
        candidate_scene_coverage: None,
        semantic_metrics_available: false,
        availability,
        diagnostics,
        facts: Vec::new(),
        character_states: HashMap::new(),
        asymmetries: Vec::new(),
        revelations: Vec::new(),
        suspense_curve: Vec::new(),
        peak_suspense_scene: 0,
        information_density: 0.0,
        manipulation_score: 0.0,
        epistemic_insights: Vec::new(),
    }
}

fn named_characters(
    sentence: &str,
    matcher: &regex::Regex,
    identities: &regex::RegexSet,
    cast: &[&str],
) -> Vec<String> {
    static WORD: OnceLock<regex::Regex> = OnceLock::new();
    let word =
        WORD.get_or_init(|| regex::Regex::new(r"^\w$").expect("constant word-character regex"));
    let mut found = BTreeSet::new();
    let mut cursor = 0;
    while let Some(hit) = matcher.find_at(sentence, cursor) {
        let is_word = |ch: char| {
            let mut bytes = [0; 4];
            word.is_match(ch.encode_utf8(&mut bytes))
        };
        let before = sentence[..hit.start()].chars().next_back();
        let after = sentence[hit.end()..].chars().next();
        if before.is_none_or(|ch| !is_word(ch)) && after.is_none_or(|ch| !is_word(ch)) {
            found.extend(identities.matches(hit.as_str()).iter());
            cursor = hit.end();
        } else {
            cursor = hit.start()
                + sentence[hit.start()..]
                    .chars()
                    .next()
                    .expect("nonempty name match")
                    .len_utf8();
        }
        if cursor >= sentence.len() {
            break;
        }
    }
    found.into_iter().map(|i| cast[i].to_string()).collect()
}

pub fn track_epistemics(
    scenes: &[&str],
    characters: &[&str],
    pov_characters: &[&str],
) -> EpistemicResult {
    track_epistemics_checked(scenes, characters, pov_characters).unwrap_or_else(|error| {
        empty_lexical_result(AnalysisAvailability::Unavailable, vec![error.to_string()])
    })
}

/// Finds bounded source-backed review candidates. Mentions, POV, testimony and
/// lexical markers do not establish truth, learning, suspense or author quality.
pub fn track_epistemics_checked(
    scenes: &[&str],
    characters: &[&str],
    pov_characters: &[&str],
) -> Result<EpistemicResult> {
    if scenes.len() > MAX_EPISTEMIC_SCENES
        || characters.len() > MAX_EPISTEMIC_CHARACTERS
        || (!pov_characters.is_empty() && pov_characters.len() != scenes.len())
    {
        return Err(Error::InvalidInput(
            "epistemic scene/cast/POV count is invalid".into(),
        ));
    }
    let mut total = 0usize;
    let mut markers = 0usize;
    for text in scenes {
        total += text.len();
        if text.len() > MAX_EPISTEMIC_TEXT_BYTES || total > MAX_EPISTEMIC_TOTAL_BYTES {
            return Err(Error::InvalidInput(
                "epistemic text exceeds byte limit".into(),
            ));
        }
        markers += text
            .chars()
            .filter(|ch| matches!(ch, '.' | '?' | '!' | '\u{2026}'))
            .take(MAX_EPISTEMIC_MARKERS + 1)
            .count();
        if markers > MAX_EPISTEMIC_MARKERS {
            return Err(Error::InvalidInput(
                "epistemic sentence marker budget exceeded".into(),
            ));
        }
    }
    let mut seen = BTreeSet::new();
    for character in characters {
        validate_name(character)?;
        if !seen.insert(character.to_lowercase()) {
            return Err(Error::InvalidInput(
                "epistemic cast contains ambiguous duplicate names".into(),
            ));
        }
    }
    for pov in pov_characters.iter().filter(|pov| !pov.is_empty()) {
        if !characters.contains(pov) {
            return Err(Error::InvalidInput(
                "epistemic POV is not a declared character".into(),
            ));
        }
    }
    let mut cast = characters.to_vec();
    cast.sort_unstable();
    let names = regex::RegexSetBuilder::new(
        cast.iter()
            .map(|name| format!(r"(?i)^{}$", regex::escape(name))),
    )
    .size_limit(8 * 1024 * 1024)
    .build()
    .map_err(|_| Error::InvalidInput("epistemic cast matcher exceeds limits".into()))?;
    let mut ordered_names = cast.clone();
    ordered_names.sort_unstable_by_key(|name| std::cmp::Reverse(name.len()));
    let pattern = ordered_names
        .iter()
        .map(|name| regex::escape(name))
        .collect::<Vec<_>>()
        .join("|");
    let name_matcher = if pattern.is_empty() {
        None
    } else {
        Some(
            regex::RegexBuilder::new(&format!("(?i)(?:{pattern})"))
                .size_limit(8 * 1024 * 1024)
                .build()
                .map_err(|_| Error::InvalidInput("epistemic name matcher exceeds limits".into()))?,
        )
    };
    let mut result = empty_lexical_result(AnalysisAvailability::Partial, vec![
        "lexical_sentence_candidates_only; truth_knowledge_and_psychological_effects_not_inferred".into(),
        "legacy_importance_is_uncalibrated_marker_tier; scalar_semantic_metrics_unavailable".into(),
    ]);
    result.character_states = cast
        .iter()
        .map(|name| {
            (
                name.to_string(),
                EpistemicState {
                    character: name.to_string(),
                    knows: HashSet::new(),
                    believes_false: HashSet::new(),
                    uncertain_about: HashSet::new(),
                },
            )
        })
        .collect();
    let mut ids = BTreeMap::new();
    let mut candidate_scenes = 0usize;
    for (scene, text) in scenes.iter().enumerate() {
        let spans = sentence_spans(text);
        result.analyzed_sentences += spans.len();
        if result.analyzed_sentences > MAX_EPISTEMIC_SENTENCES {
            return Err(Error::InvalidInput(
                "epistemic sentence count exceeds limit".into(),
            ));
        }
        let mut byte_cursor = 0;
        let mut utf16_cursor = 0u32;
        let before = result.candidates.len();
        for (start, end) in spans {
            let sentence = &text[start..end];
            if sentence.split_whitespace().take(5).count() < 5
                || sentence
                    .trim_end_matches(['"', '\'', '\u{2019}', '\u{201d}', ')', ']'])
                    .ends_with('?')
            {
                continue;
            }
            let mentioned_characters = name_matcher.as_ref().map_or_else(Vec::new, |matcher| {
                named_characters(sentence, matcher, &names, &cast)
            });
            let mut signals = Vec::new();
            if !mentioned_characters.is_empty() {
                signals.push(EpistemicSignal::NamedCharacter);
            }
            signals.extend(
                signal_matchers()
                    .iter()
                    .filter_map(|(signal, matcher)| matcher.is_match(sentence).then_some(*signal)),
            );
            if signals.is_empty() {
                continue;
            }
            let key = sentence
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase();
            let id = *ids.entry(key).or_insert_with(|| {
                let id = result.facts.len();
                let mut importance: f64 = 0.2;
                if signals.contains(&EpistemicSignal::NamedCharacter) {
                    importance += 0.3;
                }
                if signals.iter().any(|s| {
                    matches!(
                        s,
                        EpistemicSignal::Secrecy
                            | EpistemicSignal::Plan
                            | EpistemicSignal::Identity
                    )
                }) {
                    importance += 0.3;
                }
                if signals.contains(&EpistemicSignal::Relationship) {
                    importance += 0.2;
                }
                if signals.contains(&EpistemicSignal::Event) {
                    importance += 0.2;
                }
                result.facts.push(StoryFact {
                    id,
                    content: sentence.to_string(),
                    established_scene: scene,
                    importance: importance.min(1.0),
                });
                id
            });
            let utf16_start = utf16_cursor + text[byte_cursor..start].encode_utf16().count() as u32;
            let utf16_end = utf16_start + sentence.encode_utf16().count() as u32;
            byte_cursor = end;
            utf16_cursor = utf16_end;
            result.candidates.push(EpistemicCandidate {
                fact_id: id,
                scene,
                range: TextRange {
                    byte_start: start,
                    byte_end: end,
                    utf16_start,
                    utf16_end,
                },
                signals,
                mentioned_characters,
            });
        }
        candidate_scenes += usize::from(result.candidates.len() > before);
    }
    result.candidate_scene_coverage =
        (!scenes.is_empty()).then(|| candidate_scenes as f64 / scenes.len() as f64);
    result.epistemic_insights.push(format!("{} lexical candidate occurrences across {} of {} scenes; review or annotate them before drawing epistemic conclusions.", result.candidates.len(), candidate_scenes, scenes.len()));
    Ok(result)
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct AuthorEpistemicQuery {
    pub id: String,
    pub proposition_id: String,
    pub observer: AuthorAudience,
    pub subject: AuthorAudience,
    pub at_sequence: Option<u32>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthorEpistemicRelation {
    Unassessed,
    ObserverAdvantage,
    SubjectAdvantage,
    SharedCorrect,
    SharedIncorrect,
    SharedUncertainty,
    ConflictingAssertions,
    Undetermined,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthorEpistemicEndpoint {
    pub actor: AuthorAudience,
    pub assertion: AuthorAssertion,
    pub classification: AuthorKnowledgeClassification,
    pub event_id: Option<String>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthorEpistemicAnswer {
    pub query_id: String,
    pub proposition_id: String,
    pub at_sequence: Option<u32>,
    pub truth: TruthValue,
    pub truth_event_id: Option<String>,
    pub observer: AuthorEpistemicEndpoint,
    pub subject: AuthorEpistemicEndpoint,
    pub relation: AuthorEpistemicRelation,
    pub assertions_conflict: bool,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthorEpistemicResult {
    pub knowledge: AuthorKnowledgeLedger,
    pub answers: Vec<AuthorEpistemicAnswer>,
    pub assessed_queries: usize,
    pub unassessed_queries: usize,
}

fn assertion_value(assertion: AuthorAssertion) -> Option<bool> {
    match assertion {
        AuthorAssertion::Knowledge { value } | AuthorAssertion::Belief { value } => Some(value),
        _ => None,
    }
}

fn relation(
    a: AuthorKnowledgeClassification,
    b: AuthorKnowledgeClassification,
    conflict: bool,
) -> AuthorEpistemicRelation {
    use AuthorEpistemicRelation as R;
    use AuthorKnowledgeClassification as C;
    let correct = |c| matches!(c, C::Knowledge | C::CorrectBelief);
    let incorrect = |c| matches!(c, C::FalseBelief | C::ContradictedKnowledge);
    if a == C::Unassessed || b == C::Unassessed {
        R::Unassessed
    } else if correct(a) && correct(b) {
        R::SharedCorrect
    } else if correct(a) {
        R::ObserverAdvantage
    } else if correct(b) {
        R::SubjectAdvantage
    } else if incorrect(a) && incorrect(b) {
        R::SharedIncorrect
    } else if conflict {
        R::ConflictingAssertions
    } else if matches!(a, C::Unknown | C::Uncertain) && matches!(b, C::Unknown | C::Uncertain) {
        R::SharedUncertainty
    } else {
        R::Undetermined
    }
}

/// Answers only supplied comparisons at inclusive annotation checkpoints. One
/// replay order is not inferred diegetic time, reader psychology or chronology.
pub fn analyze_author_epistemics(
    manuscript: &Manuscript,
    propositions: &[AuthorProposition],
    events: &[AuthorKnowledgeEvent],
    queries: &[AuthorEpistemicQuery],
) -> Result<AuthorEpistemicResult> {
    if queries.len() > MAX_AUTHOR_EPISTEMIC_QUERIES {
        return Err(Error::InvalidInput(
            "author epistemic query count exceeds limit".into(),
        ));
    }
    let knowledge = author_knowledge::analyze_author_knowledge(manuscript, propositions, events)?;
    let ids: BTreeSet<_> = knowledge
        .propositions
        .iter()
        .map(|p| p.id.as_str())
        .collect();
    let characters: BTreeSet<_> = manuscript
        .known_characters
        .iter()
        .map(String::as_str)
        .collect();
    let mut query_ids = BTreeSet::new();
    let mut bytes = 0usize;
    for query in queries {
        validate_id(&query.id)?;
        validate_id(&query.proposition_id)?;
        if !query_ids.insert(query.id.as_str()) || !ids.contains(query.proposition_id.as_str()) {
            return Err(Error::InvalidInput(
                "author epistemic query ID duplicate or proposition missing".into(),
            ));
        }
        bytes += query.id.len() + query.proposition_id.len();
        for actor in [&query.observer, &query.subject] {
            if let AuthorAudience::Character { character_id } = actor {
                validate_id(character_id)?;
                if !characters.contains(character_id.as_str()) {
                    return Err(Error::InvalidInput(
                        "author epistemic query actor is undeclared".into(),
                    ));
                }
                bytes += character_id.len();
            }
        }
        if bytes > MAX_AUTHOR_EPISTEMIC_QUERY_BYTES {
            return Err(Error::InvalidInput(
                "author epistemic query metadata exceeds byte limit".into(),
            ));
        }
    }
    let mut truths: BTreeMap<_, _> = knowledge
        .propositions
        .iter()
        .map(|p| (p.id.as_str(), (p.initial_truth, None::<&str>)))
        .collect();
    let mut states = BTreeMap::<(AuthorAudience, &str), (AuthorAssertion, &str)>::new();
    let mut ordered: Vec<_> = queries.iter().collect();
    ordered.sort_unstable_by(|a, b| {
        a.at_sequence
            .map(u64::from)
            .unwrap_or(u64::MAX)
            .cmp(&b.at_sequence.map(u64::from).unwrap_or(u64::MAX))
            .then_with(|| a.id.cmp(&b.id))
    });
    let mut answers = Vec::with_capacity(queries.len());
    let mut next = 0;
    let mut assessed = 0;
    for query in ordered {
        let through = query.at_sequence.map(u64::from).unwrap_or(u64::MAX);
        while let Some(transition) = knowledge
            .transitions
            .get(next)
            .filter(|t| u64::from(t.sequence) <= through)
        {
            let update = match &transition.action {
                AuthorKnowledgeAction::Truth { value } => {
                    truths.insert(
                        transition.proposition_id.as_str(),
                        (*value, Some(transition.event_id.as_str())),
                    );
                    None
                }
                AuthorKnowledgeAction::Know { actor, value } => {
                    Some((actor, AuthorAssertion::Knowledge { value: *value }))
                }
                AuthorKnowledgeAction::Believe { actor, value } => {
                    Some((actor, AuthorAssertion::Belief { value: *value }))
                }
                AuthorKnowledgeAction::Uncertain { actor } => {
                    Some((actor, AuthorAssertion::Uncertain))
                }
                AuthorKnowledgeAction::Forget { actor } => Some((actor, AuthorAssertion::Unknown)),
                AuthorKnowledgeAction::Communicate {
                    recipient,
                    value,
                    uptake,
                    ..
                } => match uptake {
                    AuthorUptake::Unchanged => None,
                    AuthorUptake::Belief => {
                        Some((recipient, AuthorAssertion::Belief { value: *value }))
                    }
                    AuthorUptake::Knowledge => {
                        Some((recipient, AuthorAssertion::Knowledge { value: *value }))
                    }
                },
            };
            if let Some((actor, assertion)) = update {
                states.insert(
                    (actor.clone(), transition.proposition_id.as_str()),
                    (assertion, transition.event_id.as_str()),
                );
            }
            next += 1;
        }
        let (truth, truth_event_id) = truths[query.proposition_id.as_str()];
        let endpoint = |actor: &AuthorAudience| {
            let (assertion, event_id) = states
                .get(&(actor.clone(), query.proposition_id.as_str()))
                .map_or((AuthorAssertion::Unassessed, None), |(a, id)| {
                    (*a, Some(id.to_string()))
                });
            AuthorEpistemicEndpoint {
                actor: actor.clone(),
                assertion,
                classification: author_knowledge::classify(assertion, truth),
                event_id,
            }
        };
        let observer = endpoint(&query.observer);
        let subject = endpoint(&query.subject);
        let conflict = assertion_value(observer.assertion)
            .zip(assertion_value(subject.assertion))
            .is_some_and(|(a, b)| a != b);
        let relation = relation(observer.classification, subject.classification, conflict);
        assessed += usize::from(relation != AuthorEpistemicRelation::Unassessed);
        answers.push(AuthorEpistemicAnswer {
            query_id: query.id.clone(),
            proposition_id: query.proposition_id.clone(),
            at_sequence: query.at_sequence,
            truth,
            truth_event_id: truth_event_id.map(str::to_string),
            observer,
            subject,
            relation,
            assertions_conflict: conflict,
        });
    }
    answers.sort_unstable_by(|a, b| a.query_id.cmp(&b.query_id));
    Ok(AuthorEpistemicResult {
        knowledge,
        answers,
        assessed_queries: assessed,
        unassessed_queries: queries.len() - assessed,
    })
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
    fn test_narrated_claims_are_not_reader_knowledge() {
        let scene = "The bomb was hidden under the table. Mary sat down, unaware.";
        let result = track_epistemics(&[scene], &["Mary"], &["Mary"]);

        // Narration supplies review candidates, not verified knowledge.
        assert!(!result.facts.is_empty());
        assert!(result.character_states["Mary"].knows.is_empty());
        assert!(result.asymmetries.is_empty());
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

        assert!(!result.facts.is_empty());
        assert!(result.asymmetries.is_empty());
        assert!(!result.semantic_metrics_available);
    }

    #[test]
    fn test_lexical_candidates_do_not_fabricate_suspense() {
        let scenes = [
            "It was a quiet morning in the village.",
            "Sarah discovered that someone had been secretly watching her house.",
            "Nobody knew that the watcher planned to strike at midnight.",
            "At midnight, Sarah heard the door creak open.",
        ];
        let characters = ["Sarah"];
        let povs = ["Sarah", "Sarah", "Sarah", "Sarah"];

        let scene_refs: Vec<&str> = scenes.iter().map(|s| s.as_ref()).collect();
        let char_refs: Vec<&str> = characters.iter().map(|s| s.as_ref()).collect();
        let pov_refs: Vec<&str> = povs.iter().map(|s| s.as_ref()).collect();

        let result = track_epistemics(&scene_refs, &char_refs, &pov_refs);

        assert!(result.suspense_curve.is_empty());
        assert_eq!(result.analyzed_sentences, 4);
        assert_eq!(result.candidates.len(), 3);
        assert_eq!(result.candidate_scene_coverage, Some(0.75));
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

        assert!(result.character_states["Alice"].knows.is_empty());
        assert!(result.character_states["Bob"].knows.is_empty());
        assert!(result.revelations.is_empty());
        assert_eq!(result.candidates[0].mentioned_characters, ["Alice", "Bob"]);
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
    fn test_candidate_scene_coverage() {
        let scene1 = "John walked to the store and bought milk.";
        let scene2 = "Mary discovered the secret vault. Nobody knew the password. \
                     The treasure had been hidden for centuries.";
        let result = track_epistemics(&[scene1, scene2], &["John", "Mary"], &["John", "Mary"]);

        assert_eq!(result.candidate_scene_coverage, Some(1.0));
        assert_eq!(result.analyzed_sentences, 4);
        assert_eq!(result.information_density, 0.0);
        assert!(!result.semantic_metrics_available);
    }

    #[test]
    fn test_insights_generated() {
        let scenes = [
            "The detective knew the butler was lying about the murder weapon.",
            "The butler smiled innocently at the detective, unaware he had been found out.",
            "The detective decided to confront the butler at dinner.",
        ];
        let scene_refs: Vec<&str> = scenes.iter().map(|s| s.as_ref()).collect();
        let result = track_epistemics(
            &scene_refs,
            &["detective", "butler"],
            &["detective", "detective", "detective"],
        );

        assert!(!result.epistemic_insights.is_empty());
    }
}
