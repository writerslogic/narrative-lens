//! Pure replay of author annotations. The sequence defines one declared order,
//! not inferred story time or reader presentation. Source quotes locate an
//! annotation; neither quoted testimony nor matching truth proves knowledge.
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub use crate::structure::author_events::AuthorSource;
use crate::{
    Error, Result,
    manuscript::{Evidence, Manuscript, author_source::validate_sources, config::validate_id},
    substrate::utils::AnalysisAvailability,
};

pub const MAX_AUTHOR_PROPOSITIONS: usize = 10_000;
pub const MAX_AUTHOR_KNOWLEDGE_EVENTS: usize = 10_000;
pub const MAX_AUTHOR_KNOWLEDGE_BYTES: usize = 4 * 1024 * 1024;

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TruthValue {
    True,
    False,
    Unknown,
}

impl TruthValue {
    fn agrees(self, value: bool) -> Option<bool> {
        match self {
            Self::True => Some(value),
            Self::False => Some(!value),
            Self::Unknown => None,
        }
    }
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum AuthorAudience {
    Reader,
    Character { character_id: String },
}

impl<'de> Deserialize<'de> for AuthorAudience {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(
            tag = "kind",
            rename_all = "snake_case",
            rename_all_fields = "camelCase",
            deny_unknown_fields
        )]
        enum Wire {
            Reader {},
            Character { character_id: String },
        }
        Ok(match Wire::deserialize(deserializer)? {
            Wire::Reader {} => Self::Reader,
            Wire::Character { character_id } => Self::Character { character_id },
        })
    }
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct AuthorProposition {
    pub id: String,
    pub description: String,
    pub truth: TruthValue,
    pub evidence: AuthorSource,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthorUptake {
    Unchanged,
    Belief,
    Knowledge,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum AuthorKnowledgeAction {
    Truth {
        value: TruthValue,
    },
    Know {
        actor: AuthorAudience,
        value: bool,
    },
    Believe {
        actor: AuthorAudience,
        value: bool,
    },
    Uncertain {
        actor: AuthorAudience,
    },
    Forget {
        actor: AuthorAudience,
    },
    Communicate {
        sender: AuthorAudience,
        recipient: AuthorAudience,
        value: bool,
        uptake: AuthorUptake,
    },
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct AuthorKnowledgeEvent {
    pub id: String,
    pub sequence: u32,
    pub proposition_id: String,
    pub action: AuthorKnowledgeAction,
    pub evidence: AuthorSource,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AuthorAssertion {
    Unassessed,
    Unknown,
    Uncertain,
    Knowledge { value: bool },
    Belief { value: bool },
}

impl<'de> Deserialize<'de> for AuthorAssertion {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
        enum Wire {
            Unassessed {},
            Unknown {},
            Uncertain {},
            Knowledge { value: bool },
            Belief { value: bool },
        }
        Ok(match Wire::deserialize(deserializer)? {
            Wire::Unassessed {} => Self::Unassessed,
            Wire::Unknown {} => Self::Unknown,
            Wire::Uncertain {} => Self::Uncertain,
            Wire::Knowledge { value } => Self::Knowledge { value },
            Wire::Belief { value } => Self::Belief { value },
        })
    }
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthorKnowledgeClassification {
    Unassessed,
    Unknown,
    Uncertain,
    Knowledge,
    CorrectBelief,
    FalseBelief,
    UnverifiedBelief,
    UnverifiedKnowledge,
    ContradictedKnowledge,
}

pub(crate) fn classify(
    assertion: AuthorAssertion,
    truth: TruthValue,
) -> AuthorKnowledgeClassification {
    use AuthorKnowledgeClassification as C;
    match assertion {
        AuthorAssertion::Unassessed => C::Unassessed,
        AuthorAssertion::Unknown => C::Unknown,
        AuthorAssertion::Uncertain => C::Uncertain,
        AuthorAssertion::Knowledge { value } => match truth.agrees(value) {
            Some(true) => C::Knowledge,
            Some(false) => C::ContradictedKnowledge,
            None => C::UnverifiedKnowledge,
        },
        AuthorAssertion::Belief { value } => match truth.agrees(value) {
            Some(true) => C::CorrectBelief,
            Some(false) => C::FalseBelief,
            None => C::UnverifiedBelief,
        },
    }
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthorPropositionState {
    pub id: String,
    pub description: String,
    pub initial_truth: TruthValue,
    pub establishment: Evidence,
    pub truth: TruthValue,
    pub last_event_id: Option<String>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthorKnowledgeState {
    pub actor: AuthorAudience,
    pub proposition_id: String,
    pub assertion: AuthorAssertion,
    pub classification: AuthorKnowledgeClassification,
    pub last_event_id: Option<String>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthorKnowledgeTransition {
    pub event_id: String,
    pub sequence: u32,
    pub proposition_id: String,
    pub action: AuthorKnowledgeAction,
    pub evidence: Evidence,
    pub previous_truth: TruthValue,
    pub previous_assertion: Option<AuthorAssertion>,
    pub previous_event_id: Option<String>,
    pub sender_assertion: Option<AuthorAssertion>,
    pub sender_event_id: Option<String>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthorKnowledgeAsymmetry {
    pub proposition_id: String,
    pub character_id: String,
    pub reader_assertion: AuthorAssertion,
    pub character_assertion: AuthorAssertion,
    pub reader_classification: AuthorKnowledgeClassification,
    pub character_classification: AuthorKnowledgeClassification,
    pub reader_event_id: Option<String>,
    pub character_event_id: Option<String>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthorKnowledgeLedger {
    pub document_id: String,
    pub availability: AnalysisAvailability,
    pub diagnostics: Vec<String>,
    pub propositions: Vec<AuthorPropositionState>,
    pub states: Vec<AuthorKnowledgeState>,
    pub transitions: Vec<AuthorKnowledgeTransition>,
    pub asymmetries: Vec<AuthorKnowledgeAsymmetry>,
    pub unassessed_pairs: usize,
}

fn invalid(message: &str) -> Error {
    Error::InvalidInput(message.into())
}

fn validate_actor(actor: &AuthorAudience, characters: &BTreeSet<&str>) -> Result<usize> {
    match actor {
        AuthorAudience::Reader => Ok(0),
        AuthorAudience::Character { character_id } => {
            validate_id(character_id)?;
            if !characters.contains(character_id.as_str()) {
                return Err(invalid(
                    "author knowledge actor is not a declared character",
                ));
            }
            Ok(character_id.len())
        }
    }
}

fn charge(bytes: &mut usize, additional: usize) -> Result<()> {
    *bytes = bytes
        .checked_add(additional)
        .filter(|n| *n <= MAX_AUTHOR_KNOWLEDGE_BYTES)
        .ok_or_else(|| invalid("author knowledge metadata exceeds 4 MiB"))?;
    Ok(())
}

fn require_knowledge(truth: TruthValue, value: bool) -> Result<()> {
    if truth.agrees(value) != Some(true) {
        return Err(invalid(
            "author knowledge assertion requires matching declared known truth",
        ));
    }
    Ok(())
}

/// Replay a complete annotation snapshot. Repeated calls are idempotent; event
/// IDs/sequences within a snapshot must be unique. Replace annotations to revise
/// them. Context/time-specific propositions require distinct IDs. Knowledge is
/// an explicit author attestation, never inferred from a correct belief or quote.
pub fn analyze_author_knowledge(
    manuscript: &Manuscript,
    propositions: &[AuthorProposition],
    events: &[AuthorKnowledgeEvent],
) -> Result<AuthorKnowledgeLedger> {
    manuscript.validate()?;
    if propositions.len() > MAX_AUTHOR_PROPOSITIONS || events.len() > MAX_AUTHOR_KNOWLEDGE_EVENTS {
        return Err(invalid(
            "author knowledge exceeds 10000 propositions or 10000 events",
        ));
    }
    let characters: BTreeSet<_> = manuscript
        .known_characters
        .iter()
        .map(String::as_str)
        .collect();
    let mut ids = BTreeSet::new();
    let mut bytes = 0usize;
    let mut sources = Vec::with_capacity(propositions.len() + events.len());
    for proposition in propositions {
        validate_id(&proposition.id)?;
        if !ids.insert(proposition.id.as_str()) {
            return Err(invalid("author proposition IDs must be unique"));
        }
        if proposition.description.trim().is_empty() || proposition.description.len() > 4096 {
            return Err(invalid(
                "author proposition description requires 1..=4096 bytes",
            ));
        }
        charge(
            &mut bytes,
            proposition.id.len()
                + proposition.description.len()
                + proposition.evidence.scene_id.len()
                + proposition.evidence.quote.len(),
        )?;
        sources.push(&proposition.evidence);
    }
    let mut event_ids = BTreeSet::new();
    let mut sequences = BTreeSet::new();
    for event in events {
        validate_id(&event.id)?;
        validate_id(&event.proposition_id)?;
        if !event_ids.insert(event.id.as_str()) || !sequences.insert(event.sequence) {
            return Err(invalid(
                "author knowledge event IDs and sequences must be unique",
            ));
        }
        if !ids.contains(event.proposition_id.as_str()) {
            return Err(invalid(
                "author knowledge event refers to an unknown proposition",
            ));
        }
        let actor_bytes = match &event.action {
            AuthorKnowledgeAction::Truth { .. } => 0,
            AuthorKnowledgeAction::Know { actor, .. }
            | AuthorKnowledgeAction::Believe { actor, .. }
            | AuthorKnowledgeAction::Uncertain { actor }
            | AuthorKnowledgeAction::Forget { actor } => validate_actor(actor, &characters)?,
            AuthorKnowledgeAction::Communicate {
                sender, recipient, ..
            } => validate_actor(sender, &characters)? + validate_actor(recipient, &characters)?,
        };
        charge(
            &mut bytes,
            event.id.len()
                + event.proposition_id.len()
                + actor_bytes
                + event.evidence.scene_id.len()
                + event.evidence.quote.len(),
        )?;
        sources.push(&event.evidence);
    }
    let evidence = validate_sources(manuscript, &sources)?;
    let mut facts: BTreeMap<_, _> = propositions
        .iter()
        .zip(&evidence)
        .map(|(p, evidence)| {
            (
                p.id.clone(),
                AuthorPropositionState {
                    id: p.id.clone(),
                    description: p.description.clone(),
                    initial_truth: p.truth,
                    establishment: evidence.clone(),
                    truth: p.truth,
                    last_event_id: None,
                },
            )
        })
        .collect();
    let mut ordered: Vec<_> = events
        .iter()
        .zip(evidence.into_iter().skip(propositions.len()))
        .collect();
    ordered.sort_unstable_by_key(|(event, _)| event.sequence);
    let mut current: BTreeMap<(AuthorAudience, String), (AuthorAssertion, String)> =
        BTreeMap::new();
    let mut transitions = Vec::with_capacity(events.len());
    for (event, evidence) in ordered {
        let fact = facts
            .get_mut(&event.proposition_id)
            .expect("proposition references validated");
        let mut transition = AuthorKnowledgeTransition {
            event_id: event.id.clone(),
            sequence: event.sequence,
            proposition_id: event.proposition_id.clone(),
            action: event.action.clone(),
            evidence,
            previous_truth: fact.truth,
            previous_assertion: None,
            previous_event_id: fact.last_event_id.clone(),
            sender_assertion: None,
            sender_event_id: None,
        };
        let update = match &event.action {
            AuthorKnowledgeAction::Truth { value } => {
                fact.truth = *value;
                fact.last_event_id = Some(event.id.clone());
                None
            }
            AuthorKnowledgeAction::Know { actor, value } => {
                require_knowledge(fact.truth, *value)?;
                Some((actor, Some(AuthorAssertion::Knowledge { value: *value })))
            }
            AuthorKnowledgeAction::Believe { actor, value } => {
                Some((actor, Some(AuthorAssertion::Belief { value: *value })))
            }
            AuthorKnowledgeAction::Uncertain { actor } => {
                Some((actor, Some(AuthorAssertion::Uncertain)))
            }
            AuthorKnowledgeAction::Forget { actor } => {
                Some((actor, Some(AuthorAssertion::Unknown)))
            }
            AuthorKnowledgeAction::Communicate {
                sender,
                recipient,
                value,
                uptake,
            } => {
                let sender_state = current.get(&(sender.clone(), event.proposition_id.clone()));
                transition.sender_assertion =
                    Some(sender_state.map_or(AuthorAssertion::Unassessed, |s| s.0));
                transition.sender_event_id = sender_state.map(|s| s.1.clone());
                let assertion = match uptake {
                    AuthorUptake::Unchanged => None,
                    AuthorUptake::Belief => Some(AuthorAssertion::Belief { value: *value }),
                    AuthorUptake::Knowledge => {
                        require_knowledge(fact.truth, *value)?;
                        Some(AuthorAssertion::Knowledge { value: *value })
                    }
                };
                Some((recipient, assertion))
            }
        };
        if let Some((actor, assertion)) = update {
            let key = (actor.clone(), event.proposition_id.clone());
            let previous = current.get(&key);
            transition.previous_assertion =
                Some(previous.map_or(AuthorAssertion::Unassessed, |s| s.0));
            transition.previous_event_id = previous.map(|s| s.1.clone());
            if let Some(assertion) = assertion {
                current.insert(key, (assertion, event.id.clone()));
            }
        }
        transitions.push(transition);
    }
    let mut states = Vec::with_capacity(current.len());
    let mut asymmetries = Vec::new();
    for ((actor, proposition_id), (assertion, event_id)) in &current {
        let truth = facts[proposition_id].truth;
        let classification = classify(*assertion, truth);
        states.push(AuthorKnowledgeState {
            actor: actor.clone(),
            proposition_id: proposition_id.clone(),
            assertion: *assertion,
            classification,
            last_event_id: Some(event_id.clone()),
        });
        if let AuthorAudience::Character { character_id } = actor {
            let reader = current.get(&(AuthorAudience::Reader, proposition_id.clone()));
            let reader_assertion = reader.map_or(AuthorAssertion::Unassessed, |s| s.0);
            if reader_assertion != *assertion {
                asymmetries.push(AuthorKnowledgeAsymmetry {
                    proposition_id: proposition_id.clone(),
                    character_id: character_id.clone(),
                    reader_assertion,
                    character_assertion: *assertion,
                    reader_classification: classify(reader_assertion, truth),
                    character_classification: classification,
                    reader_event_id: reader.map(|s| s.1.clone()),
                    character_event_id: Some(event_id.clone()),
                });
            }
        }
    }
    let total_pairs = propositions.len() * (characters.len() + 1);
    Ok(AuthorKnowledgeLedger {
        document_id: manuscript.document_id.clone(), availability: AnalysisAvailability::Complete,
        diagnostics: vec!["Author annotations only; quotes locate annotations without proving entailment. One declared replay order does not reconstruct story/presentation chronology. Omitted actor/proposition pairs are unassessed, not ignorant or false. Asymmetries compare supplied current assertions, not suspense or information bits.".into()],
        propositions: facts.into_values().collect(), unassessed_pairs: total_pairs - states.len(), states, transitions, asymmetries,
    })
}
