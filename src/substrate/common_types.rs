use serde::{Deserialize, Serialize};

/// A character-scene coordinate used across multiple analysis modules.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CharacterScenePoint {
    pub character: String,
    pub scene: usize,
}

/// Writer-facing feedback note with scene location and actionable suggestion.
/// Standardized across all V4.0 hard-science framework modules.
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoteSeverity {
    Info,
    Warning,
    Critical,
}

/// UTF-8-safe substring extraction. Returns a slice from approximately `start` bytes
/// to approximately `end` bytes, adjusted to char boundaries.
pub fn safe_substr(s: &str, start: usize, end: usize) -> &str {
    let start = s
        .char_indices()
        .map(|(i, _)| i)
        .take_while(|&i| i <= start)
        .last()
        .unwrap_or(0);
    let end = s
        .char_indices()
        .map(|(i, c)| i + c.len_utf8())
        .take_while(|&i| i <= end)
        .last()
        .unwrap_or(s.len())
        .min(s.len());
    &s[start..end]
}

/// Where a field's value came from. Re-extraction may overwrite [`AiDerived`]
/// values but never [`UserEdited`] ones — the reconciliation rule that keeps a
/// writer's manual corrections from being clobbered.
///
/// [`AiDerived`]: Provenance::AiDerived
/// [`UserEdited`]: Provenance::UserEdited
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    #[default]
    AiDerived,
    UserEdited,
}
