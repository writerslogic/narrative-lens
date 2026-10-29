use super::finding::Severity;
pub use super::rules::{RULES, RuleDefinition, RuleId, RuleMethod, rule_definitions};
use crate::{
    Error, Result,
    continuity::entity::EntityCorrection,
    structure::genre::{Genre, get_genre_profile},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "camelCase")]
pub struct Thresholds {
    pub max_grade: Option<f64>,
    pub max_sentence_words: Option<f64>,
    pub min_scene_words: Option<usize>,
    pub min_structure_scenes: Option<usize>,
    pub nli_contradiction: Option<f64>,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Intention {
    Repetition,
    Dialect,
    ComplexProse,
    LongSentences,
    Nonlinear,
    OpenEnding,
    Supernatural,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Suppression {
    #[cfg_attr(feature = "bindings", ts(type = "RuleId | null"))]
    pub rule_id: Option<String>,
    pub scene_id: Option<String>,
    pub fingerprint: Option<String>,
    pub reason: String,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelPolicy {
    #[default]
    Disabled,
    Fallback,
    Required,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "camelCase")]
pub struct AnalysisLimits {
    pub max_findings: usize,
    pub scene_candidates: usize,
    pub nli_pairs: usize,
    pub nli_sentences_per_scene: usize,
    pub nli_window_scenes: usize,
    pub model_input_bytes: usize,
}
impl Default for AnalysisLimits {
    fn default() -> Self {
        Self {
            max_findings: 10_000,
            scene_candidates: 4096,
            nli_pairs: 128,
            nli_sentences_per_scene: 8,
            nli_window_scenes: 1,
            model_input_bytes: 16 * 1024,
        }
    }
}
impl AnalysisLimits {
    fn validate(&self) -> Result<()> {
        if !(1..=100_000).contains(&self.max_findings)
            || !(1..=65_536).contains(&self.scene_candidates)
            || !(1..=65_536).contains(&self.nli_pairs)
            || !(1..=64).contains(&self.nli_sentences_per_scene)
            || !(1..=128).contains(&self.nli_window_scenes)
            || !(1..=65_536).contains(&self.model_input_bytes)
        {
            return Err(Error::InvalidInput(
                "analysis limits outside documented bounds".into(),
            ));
        }
        Ok(())
    }
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "camelCase")]
pub struct Configuration {
    pub genre: Genre,
    pub limits: AnalysisLimits,
    pub thresholds: Thresholds,
    pub dictionary: Vec<String>,
    /// Replacement list of whole-word editorial terms; empty by default.
    pub lexicon: Vec<String>,
    pub entity_corrections: Vec<EntityCorrection>,
    /// None selects all supported rules; Some(empty) selects none.
    #[cfg_attr(feature = "bindings", ts(type = "Array<RuleId> | null"))]
    pub rules: Option<BTreeSet<String>>,
    pub suppressions: Vec<Suppression>,
    pub minimum_severity: Severity,
    pub structure_template: String,
    pub chronology: bool,
    pub nli: ModelPolicy,
    /// Caller changes this when replacing an injected model/backend.
    pub model_revision: String,
}
impl Default for Configuration {
    fn default() -> Self {
        Self {
            genre: Genre::General,
            limits: AnalysisLimits::default(),
            thresholds: Thresholds::default(),
            dictionary: Vec::new(),
            lexicon: Vec::new(),
            entity_corrections: Vec::new(),
            rules: None,
            suppressions: Vec::new(),
            minimum_severity: Severity::Info,
            structure_template: "three_act".into(),
            chronology: false,
            nli: ModelPolicy::Disabled,
            model_revision: String::new(),
        }
    }
}
impl Configuration {
    pub fn validate(&self) -> Result<()> {
        self.limits.validate()?;
        let valid = |value: Option<f64>, min: f64, max: f64| {
            value.is_none_or(|v| v.is_finite() && (min..=max).contains(&v))
        };
        if !valid(self.thresholds.max_grade, 0.0, 30.0)
            || !valid(self.thresholds.max_sentence_words, 1.0, 500.0)
            || !valid(self.thresholds.nli_contradiction, 0.0, 1.0)
            || self
                .thresholds
                .min_scene_words
                .is_some_and(|n| !(1..=100_000).contains(&n))
            || self
                .thresholds
                .min_structure_scenes
                .is_some_and(|n| !(2..=10_000).contains(&n))
        {
            return Err(Error::InvalidInput(
                "threshold outside documented bounds".into(),
            ));
        }
        crate::structure::structure::get_structure_template(&self.structure_template)?;
        for list in [&self.dictionary, &self.lexicon] {
            if list.len() > 4096
                || list
                    .iter()
                    .any(|s| s.trim().is_empty() || s.len() > 128 || s.contains(['\n', '\r', '\0']))
            {
                return Err(Error::InvalidInput("dictionary/lexicon requires at most 4096 nonempty entries of at most 128 bytes".into()));
            }
        }
        if self.entity_corrections.len() > 4096
            || self.suppressions.len() > 4096
            || self.model_revision.len() > 256
        {
            return Err(Error::InvalidInput(
                "configuration exceeds collection/string limits".into(),
            ));
        }
        for c in &self.entity_corrections {
            validate_name(&c.surface)?;
            if let crate::continuity::entity::CorrectionAction::Assign { canonical } = &c.action {
                validate_name(canonical)?;
            }
        }
        if self
            .rules
            .as_ref()
            .is_some_and(|set| set.iter().any(|r| r.parse::<RuleId>().is_err()))
        {
            return Err(Error::InvalidInput("unknown rule ID".into()));
        }
        if self
            .rules
            .as_ref()
            .is_some_and(|rules| rules.contains("NL-GRAMMAR-CAPITALIZATION"))
        {
            return Err(Error::InvalidInput(
                "capitalization is an adapter-only rule; no facade detector is available".into(),
            ));
        }
        for s in &self.suppressions {
            if s.rule_id.is_none() && s.fingerprint.is_none()
                || s.rule_id
                    .as_ref()
                    .is_some_and(|r| r.parse::<RuleId>().is_err())
                || s.reason.trim().is_empty()
                || s.reason.len() > 1024
                || s.fingerprint.as_ref().is_some_and(|s| s.len() > 128)
            {
                return Err(Error::InvalidInput(
                    "suppression requires a valid rule or fingerprint and a reason".into(),
                ));
            }
            if let Some(id) = &s.scene_id {
                validate_id(id)?;
            }
        }
        if self.nli == ModelPolicy::Required && !self.selected("NL-CONTINUITY-NLI") {
            return Err(Error::InvalidInput(
                "required NLI conflicts with unselected NLI rule".into(),
            ));
        }
        Ok(())
    }
    pub fn selected(&self, rule: &str) -> bool {
        self.rules.as_ref().is_none_or(|r| r.contains(rule))
    }
    pub fn max_grade(&self) -> f64 {
        self.thresholds
            .max_grade
            .unwrap_or_else(|| get_genre_profile(self.genre).target_fkgl_max)
    }
    pub fn max_sentence_words(&self) -> f64 {
        self.thresholds
            .max_sentence_words
            .unwrap_or_else(|| get_genre_profile(self.genre).target_velocity_max)
    }
    pub fn min_scene_words(&self) -> usize {
        self.thresholds.min_scene_words.unwrap_or(100)
    }
}
pub(crate) fn validate_id(id: &str) -> Result<()> {
    if id.trim().is_empty() || id.len() > 256 || id.chars().any(char::is_control) {
        return Err(Error::InvalidInput(
            "IDs must contain 1..=256 bytes and no control characters".into(),
        ));
    }
    Ok(())
}
pub(crate) fn validate_name(name: &str) -> Result<()> {
    if name.trim().is_empty() || name.len() > 128 || name.chars().any(char::is_control) {
        return Err(Error::InvalidInput(
            "character names must contain 1..=128 bytes and no control characters".into(),
        ));
    }
    Ok(())
}
