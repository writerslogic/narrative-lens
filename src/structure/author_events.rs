use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    Error, Result,
    manuscript::{Evidence, Manuscript, author_source::validate_sources, config::validate_id},
    substrate::utils::AnalysisAvailability,
};

pub const MAX_AUTHOR_EVENTS: usize = 10_000;
pub const MAX_AUTHOR_EVENT_BYTES: usize = 4 * 1024 * 1024;

pub use crate::manuscript::author_source::AuthorSource;

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthorResolution {
    Fulfilled,
    Partial,
    Subverted,
    Abandoned,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum AuthorPromiseEvent {
    Setup {
        id: String,
        description: String,
        evidence: AuthorSource,
    },
    Resolution {
        id: String,
        setup_id: String,
        outcome: AuthorResolution,
        evidence: AuthorSource,
    },
}

impl AuthorPromiseEvent {
    fn id(&self) -> &str {
        match self {
            Self::Setup { id, .. } | Self::Resolution { id, .. } => id,
        }
    }
    fn source(&self) -> &AuthorSource {
        match self {
            Self::Setup { evidence, .. } | Self::Resolution { evidence, .. } => evidence,
        }
    }
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfirmedResolution {
    pub event_id: String,
    pub outcome: AuthorResolution,
    pub evidence: Evidence,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthorPromise {
    pub id: String,
    pub description: String,
    pub setup: Evidence,
    /// None means resolution is unknown, including when later prose appears to answer a question.
    pub resolution: Option<ConfirmedResolution>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthorPromiseLedger {
    pub document_id: String,
    pub availability: AnalysisAvailability,
    pub diagnostics: Vec<String>,
    pub promises: Vec<AuthorPromise>,
    pub unknown_count: usize,
    pub fulfilled_count: usize,
    pub partial_count: usize,
    pub subverted_count: usize,
    pub abandoned_count: usize,
}

fn invalid(message: &str) -> Error {
    Error::InvalidInput(message.into())
}

/// Validate a snapshot of author annotations against exact original scenes.
/// IDs identify annotations across edits/reordering; source ranges must be refreshed after edits.
/// Each setup has at most one current resolution; replace that event in the next snapshot to revise it.
pub fn analyze_promise_events(
    manuscript: &Manuscript,
    events: &[AuthorPromiseEvent],
) -> Result<AuthorPromiseLedger> {
    manuscript.validate()?;
    if events.len() > MAX_AUTHOR_EVENTS {
        return Err(invalid("author event count exceeds 10000"));
    }
    let mut ids = BTreeSet::new();
    let mut setup_ids = BTreeSet::new();
    let mut resolutions = BTreeSet::new();
    let mut metadata_bytes = 0usize;
    for event in events {
        validate_id(event.id())?;
        if !ids.insert(event.id()) {
            return Err(invalid("author event IDs must be unique"));
        }
        let source = event.source();
        let additional = match event {
            AuthorPromiseEvent::Setup {
                id, description, ..
            } => {
                if description.trim().is_empty() || description.len() > 4096 {
                    return Err(invalid("author setup description requires 1..=4096 bytes"));
                }
                setup_ids.insert(id.as_str());
                description.len()
            }
            AuthorPromiseEvent::Resolution { setup_id, .. } => {
                validate_id(setup_id)?;
                if !resolutions.insert(setup_id.as_str()) {
                    return Err(invalid("each setup permits one current author resolution"));
                }
                setup_id.len()
            }
        };
        metadata_bytes = metadata_bytes
            .checked_add(event.id().len() + source.scene_id.len() + source.quote.len() + additional)
            .filter(|bytes| *bytes <= MAX_AUTHOR_EVENT_BYTES)
            .ok_or_else(|| invalid("author event metadata exceeds 4 MiB"))?;
    }
    if resolutions.iter().any(|id| !setup_ids.contains(id)) {
        return Err(invalid("author resolution refers to an unknown setup"));
    }
    let evidence = validate_sources(
        manuscript,
        &events
            .iter()
            .map(AuthorPromiseEvent::source)
            .collect::<Vec<_>>(),
    )?;
    let mut promises = BTreeMap::new();
    let mut confirmed = BTreeMap::new();
    for (event, evidence) in events.iter().zip(evidence) {
        match event {
            AuthorPromiseEvent::Setup {
                id, description, ..
            } => {
                promises.insert(
                    id,
                    AuthorPromise {
                        id: id.clone(),
                        description: description.clone(),
                        setup: evidence,
                        resolution: None,
                    },
                );
            }
            AuthorPromiseEvent::Resolution {
                id,
                setup_id,
                outcome,
                ..
            } => {
                confirmed.insert(
                    setup_id,
                    ConfirmedResolution {
                        event_id: id.clone(),
                        outcome: *outcome,
                        evidence,
                    },
                );
            }
        }
    }
    for (setup_id, resolution) in confirmed {
        promises
            .get_mut(setup_id)
            .expect("setup reference validated")
            .resolution = Some(resolution);
    }
    let mut ledger = AuthorPromiseLedger {
        document_id: manuscript.document_id.clone(),
        availability: AnalysisAvailability::Complete,
        diagnostics: vec![
            "Author-confirmed annotations only; unlinked resolutions remain unknown.".into(),
        ],
        promises: promises.into_values().collect(),
        unknown_count: 0,
        fulfilled_count: 0,
        partial_count: 0,
        subverted_count: 0,
        abandoned_count: 0,
    };
    for promise in &ledger.promises {
        match promise
            .resolution
            .as_ref()
            .map(|resolution| resolution.outcome)
        {
            None => ledger.unknown_count += 1,
            Some(AuthorResolution::Fulfilled) => ledger.fulfilled_count += 1,
            Some(AuthorResolution::Partial) => ledger.partial_count += 1,
            Some(AuthorResolution::Subverted) => ledger.subverted_count += 1,
            Some(AuthorResolution::Abandoned) => ledger.abandoned_count += 1,
        }
    }
    Ok(ledger)
}
