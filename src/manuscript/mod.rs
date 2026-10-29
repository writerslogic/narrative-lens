//! Optional orchestration over the standalone analyzers; no I/O or persistence.
pub(crate) mod author_source;
mod collector;
pub mod comparison;
pub mod config;
mod context;
mod continuity;
pub mod finding;
pub mod rules;
mod validation;

use crate::{Error, Result, onnx::InferenceBackend};
pub use comparison::{
    ChangeDirection, MatchKind, PersistentFinding, RevisionComparison, UnassessedFinding,
    assign_fingerprints, compare_reports, try_compare_reports,
};
pub use config::{
    AnalysisLimits, Configuration, Intention, ModelPolicy, RULES, RuleDefinition, RuleId,
    RuleMethod, Suppression, Thresholds, rule_definitions,
};
pub use context::{AnalysisContext, CacheStats};
pub use continuity::{ChronologyReport, ReaderKnowledge};
pub use finding::{
    Confidence, ConfidenceSource, Evidence, EvidenceContext, EvidenceRole, Finding, Scope,
    Severity, TextRange,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Chronology {
    pub key: String,
    pub lane: String,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Scene {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub chronology: Option<Chronology>,
    #[serde(default)]
    pub chronology_lane: Option<String>,
    #[serde(default)]
    pub intentions: Vec<Intention>,
    #[serde(default)]
    pub purpose: String,
}
impl Scene {
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            chronology: None,
            chronology_lane: None,
            intentions: Vec::new(),
            purpose: String::new(),
        }
    }
    #[deprecated(
        since = "0.2.0",
        note = "use `Scene::new` and set `chronology` / `chronology_lane` from the timeline entry directly"
    )]
    pub fn from_timeline(
        entry: &crate::continuity::timeline::TimelineEntry,
        text: impl Into<String>,
    ) -> Self {
        let mut scene = Self::new(&entry.document_id, text);
        scene.chronology_lane = entry.lane.clone();
        if let Some(key) = &entry.chronology_key {
            scene.chronology = Some(Chronology {
                key: key.clone(),
                lane: entry.lane.clone().unwrap_or_else(|| "main".into()),
            });
        }
        scene
    }
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Manuscript {
    pub document_id: String,
    pub scenes: Vec<Scene>,
    #[serde(default)]
    pub known_characters: Vec<String>,
    #[serde(default)]
    pub configuration: Configuration,
}
impl Manuscript {
    pub fn validate(&self) -> Result<()> {
        config::validate_id(&self.document_id)?;
        self.configuration.validate()?;
        if self.scenes.len() > 10_000 || self.known_characters.len() > 256 {
            return Err(Error::InvalidInput(
                "manuscript exceeds 10000 scenes or 256 known characters".into(),
            ));
        }
        let mut ids = BTreeSet::new();
        let mut bytes = 0usize;
        for scene in &self.scenes {
            config::validate_id(&scene.id)?;
            if !ids.insert(&scene.id) {
                return Err(Error::InvalidInput("scene IDs must be unique".into()));
            }
            bytes = bytes
                .checked_add(scene.text.len())
                .ok_or_else(|| Error::InvalidInput("manuscript length overflow".into()))?;
            if scene.text.len() > 16 * 1024 * 1024
                || bytes > 64 * 1024 * 1024
                || scene.purpose.len() > 4096
                || scene.intentions.len() > 32
            {
                return Err(Error::InvalidInput(
                    "manuscript exceeds text or metadata limits".into(),
                ));
            }
            if let Some(lane) = &scene.chronology_lane {
                config::validate_id(lane)?;
            }
            if scene
                .chronology
                .as_ref()
                .zip(scene.chronology_lane.as_ref())
                .is_some_and(|(c, lane)| &c.lane != lane)
            {
                return Err(Error::InvalidInput(
                    "chronology lane conflicts with its lane hint".into(),
                ));
            }
            if let Some(c) = &scene.chronology {
                config::validate_id(&c.key)?;
                config::validate_id(&c.lane)?;
            }
        }
        for c in &self.known_characters {
            config::validate_name(c)?;
        }
        if self
            .known_characters
            .iter()
            .map(|s| s.to_lowercase())
            .collect::<BTreeSet<_>>()
            .len()
            != self.known_characters.len()
        {
            return Err(Error::InvalidInput(
                "known characters must be unique ignoring case".into(),
            ));
        }
        for suppression in &self.configuration.suppressions {
            if suppression
                .scene_id
                .as_ref()
                .is_some_and(|id| !ids.contains(id))
            {
                return Err(Error::InvalidInput(
                    "suppression refers to an unknown scene ID".into(),
                ));
            }
        }
        Ok(())
    }
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Complete,
    HeuristicFallback,
    Unavailable,
    InsufficientInput,
    ExecutionFailure,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Coverage {
    pub analyzer: String,
    pub status: Status,
    pub total_units: usize,
    pub analyzed_units: usize,
    pub unit: String,
    pub diagnostics: Vec<String>,
}
impl Coverage {
    /// Scene-unit coverage: a selected analyzer saw `analyzed` of `eligible`
    /// nonempty scenes out of `total`; diagnostics start empty.
    pub(crate) fn scenes(
        analyzer: &str,
        selected: bool,
        analyzed: usize,
        eligible: usize,
        total: usize,
    ) -> Self {
        Self {
            analyzer: analyzer.into(),
            status: if !selected {
                Status::Unavailable
            } else if analyzed == 0 {
                Status::InsufficientInput
            } else if analyzed < eligible {
                Status::HeuristicFallback
            } else {
                Status::Complete
            },
            total_units: total,
            analyzed_units: if selected { analyzed } else { 0 },
            unit: "scenes".into(),
            diagnostics: Vec::new(),
        }
    }
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Diagnostic {
    pub code: String,
    pub message: String,
    pub scene_ids: Vec<String>,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneMetrics {
    pub scene_id: String,
    pub word_count: usize,
    pub sentence_count: usize,
    pub grade: f64,
    pub sentence_words: f64,
    pub dialogue_ratio: f64,
    pub tension: f64,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SuppressedFinding {
    pub finding: Finding,
    pub reason: String,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Report {
    #[serde(default)]
    pub summary: ReportSummary,
    pub schema_version: String,
    #[serde(default)]
    pub configuration: Option<Configuration>,
    #[serde(default)]
    pub model_identities: Vec<crate::onnx::model::ModelIdentity>,
    pub document_id: String,
    /// Presentation order, never changed by chronology.
    pub scene_ids: Vec<String>,
    pub findings: Vec<Finding>,
    pub suppressed: Vec<SuppressedFinding>,
    pub metrics: Vec<SceneMetrics>,
    pub coverage: Vec<Coverage>,
    pub diagnostics: Vec<Diagnostic>,
    pub chronology: ChronologyReport,
}

pub fn analyze_manuscript(manuscript: &Manuscript) -> Result<Report> {
    AnalysisContext::default().analyze(manuscript)
}
pub fn analyze_with_backend(
    manuscript: &Manuscript,
    backend: &dyn InferenceBackend,
) -> Result<Report> {
    AnalysisContext::default().analyze_with_backend(manuscript, backend)
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReportSummary {
    pub total_findings: usize,
    pub suppressed_findings: usize,
    pub returned_findings: usize,
    pub omitted_findings: usize,
    pub omitted_suppressed: usize,
    pub severity_counts: std::collections::BTreeMap<Severity, usize>,
}
