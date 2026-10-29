use serde::{Deserialize, Serialize};

/// A character-scene coordinate used across multiple analysis modules.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CharacterScenePoint {
    pub character: String,
    pub scene: usize,
}

/// Writer-facing feedback note with scene location and actionable suggestion.
/// Standardized across all V4.0 hard-science framework modules.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WriterNote {
    pub scene: usize,
    pub scene_end: Option<usize>,
    pub category: String,
    pub observation: String,
    pub reader_impact: String,
    pub suggestion: String,
}

/// Severity of a writer note, used for triage prioritization.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoteSeverity {
    Info,
    Warning,
    Critical,
}

pub(crate) fn invalid(message: &str) -> crate::Error {
    crate::Error::InvalidInput(message.into())
}

/// Where a field's value came from. Re-extraction may overwrite [`AiDerived`]
/// values but never [`UserEdited`] ones — the reconciliation rule that keeps a
/// writer's manual corrections from being clobbered.
///
/// [`AiDerived`]: Provenance::AiDerived
/// [`UserEdited`]: Provenance::UserEdited
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    #[default]
    AiDerived,
    UserEdited,
}
