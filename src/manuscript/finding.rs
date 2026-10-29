use crate::{Error, Result};
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Suggestion,
    Warning,
    Error,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextRange {
    pub byte_start: usize,
    pub byte_end: usize,
    pub utf16_start: u32,
    pub utf16_end: u32,
}
impl TextRange {
    /// Zero-based, half-open offsets into the exact original scene text.
    pub fn new(text: &str, start: usize, end: usize) -> Result<Self> {
        if start > end
            || end > text.len()
            || !text.is_char_boundary(start)
            || !text.is_char_boundary(end)
        {
            return Err(Error::InvalidInput(
                "range must follow UTF-8 boundaries within the original text".into(),
            ));
        }
        let utf16_start = u32::try_from(text[..start].encode_utf16().count())
            .map_err(|_| Error::InvalidInput("text exceeds UTF-16 offset limit".into()))?;
        let utf16_end = utf16_start
            .checked_add(
                u32::try_from(text[start..end].encode_utf16().count())
                    .map_err(|_| Error::InvalidInput("range exceeds UTF-16 offset limit".into()))?,
            )
            .ok_or_else(|| Error::InvalidInput("range exceeds UTF-16 offset limit".into()))?;
        Ok(Self {
            byte_start: start,
            byte_end: end,
            utf16_start,
            utf16_end,
        })
    }
    pub fn from_utf16(text: &str, start: u32, end: u32) -> Result<Self> {
        let byte = |target: u32| {
            let mut units = 0;
            for (offset, ch) in text.char_indices() {
                if units == target {
                    return Some(offset);
                }
                units += ch.len_utf16() as u32;
            }
            (units == target).then_some(text.len())
        };
        Self::new(
            text,
            byte(start).ok_or_else(|| {
                Error::InvalidInput("UTF-16 start splits a surrogate pair or exceeds text".into())
            })?,
            byte(end).ok_or_else(|| {
                Error::InvalidInput("UTF-16 end splits a surrogate pair or exceeds text".into())
            })?,
        )
    }
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceRole {
    Primary,
    Conflict,
    Context,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceContext {
    pub before: String,
    pub after: String,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Evidence {
    pub document_id: String,
    pub scene_id: String,
    pub range: TextRange,
    pub quote: String,
    pub role: EvidenceRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<EvidenceContext>,
}
impl Evidence {
    pub fn new(
        document: &str,
        scene: &str,
        text: &str,
        start: usize,
        end: usize,
        role: EvidenceRole,
    ) -> Result<Self> {
        let range = TextRange::new(text, start, end)?;
        Ok(Self {
            document_id: document.into(),
            scene_id: scene.into(),
            range,
            quote: text[start..end].into(),
            role,
            context: Some(context(text, start, end)),
        })
    }
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum Scope {
    Passage { scene_id: String },
    Scene { scene_id: String },
    Manuscript { scene_ids: Vec<String> },
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceSource {
    Heuristic,
    Model,
    Calibrated,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Confidence {
    pub score: f64,
    pub source: ConfidenceSource,
    pub method: String,
    pub calibration_id: Option<String>,
}
impl Confidence {
    pub fn heuristic(score: f64, method: &str) -> Self {
        assert!(score.is_finite(), "confidence must be finite");
        Self {
            score: score.clamp(0.0, 1.0),
            source: ConfidenceSource::Heuristic,
            method: method.into(),
            calibration_id: None,
        }
    }
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Finding {
    pub document_id: String,
    #[cfg_attr(feature = "bindings", ts(type = "RuleId"))]
    pub rule_id: String,
    pub analyzer: String,
    pub severity: Severity,
    pub message: String,
    pub suggestion: String,
    pub confidence: Confidence,
    pub scope: Scope,
    /// Primary passage first; contradictory passages carry the Conflict role.
    /// Aggregate findings may have no pinpoint evidence; their scope is explicit.
    pub evidence: Vec<Evidence>,
    pub fingerprint: String,
}

/// Adapts the existing UTF-16 grammar API without changing its result types.
pub fn adapt_grammar(
    document: &str,
    scene: &str,
    text: &str,
    legacy: &crate::craft::grammar::GrammarFinding,
) -> Result<Finding> {
    super::config::validate_id(document)?;
    super::config::validate_id(scene)?;
    let offsets = crate::substrate::offsets::OffsetIndex::new(text)?;
    let mut finding = adapt_grammar_indexed(document, scene, &offsets, legacy)?;
    super::assign_fingerprints(document, std::slice::from_mut(&mut finding))?;
    Ok(finding)
}

pub(crate) fn adapt_grammar_indexed(
    document: &str,
    scene: &str,
    offsets: &crate::substrate::offsets::OffsetIndex<'_>,
    legacy: &crate::craft::grammar::GrammarFinding,
) -> Result<Finding> {
    use crate::craft::grammar::FindingKind;
    let text = offsets.text();
    let (rule, severity, score) = match legacy.kind {
        FindingKind::Repetition => ("NL-GRAMMAR-REPETITION", Severity::Warning, 0.8),
        FindingKind::Whitespace => ("NL-GRAMMAR-WHITESPACE", Severity::Info, 0.8),
        FindingKind::Spelling => ("NL-GRAMMAR-SPELLING", Severity::Suggestion, 0.5),
        FindingKind::Capitalization => ("NL-GRAMMAR-CAPITALIZATION", Severity::Suggestion, 0.8),
    };
    let start = offsets
        .byte(legacy.utf16_start)
        .ok_or_else(|| Error::InvalidInput("invalid grammar UTF-16 start".into()))?;
    let end = offsets
        .byte(legacy.utf16_end)
        .ok_or_else(|| Error::InvalidInput("invalid grammar UTF-16 end".into()))?;
    if start > end {
        return Err(Error::InvalidInput("grammar range is reversed".into()));
    }
    let range = TextRange {
        byte_start: start,
        byte_end: end,
        utf16_start: legacy.utf16_start,
        utf16_end: legacy.utf16_end,
    };
    Ok(Finding {
        document_id: document.into(),
        rule_id: rule.into(),
        analyzer: "grammar".into(),
        severity,
        message: legacy.message.clone(),
        suggestion: legacy.replacements.join(", "),
        confidence: Confidence::heuristic(
            score,
            "legacy-grammar-v1; dictionary absence is not proof of misspelling",
        ),
        scope: Scope::Passage {
            scene_id: scene.into(),
        },
        evidence: vec![Evidence {
            document_id: document.into(),
            scene_id: scene.into(),
            quote: text[start..end].into(),
            role: EvidenceRole::Primary,
            context: Some(context(text, start, end)),
            range,
        }],
        fingerprint: String::new(),
    })
}

/// Adapter for legacy explanations that have no source span. Never invents one.
pub fn adapt_explanation(
    document_id: &str,
    rule_id: &str,
    scene_ids: Vec<String>,
    legacy: &crate::substrate::explain::ExplainedFinding,
) -> Finding {
    let mut finding = Finding {
        document_id: document_id.into(),
        rule_id: rule_id.into(),
        analyzer: legacy.finding_type.clone(),
        severity: match legacy.severity.as_str() {
            "warning" => Severity::Warning,
            "error" => Severity::Error,
            "info" => Severity::Info,
            _ => Severity::Suggestion,
        },
        message: legacy.explanation.clone(),
        suggestion: legacy.suggestion.clone(),
        confidence: Confidence::heuristic(legacy.confidence, "legacy-explanation-v1; uncalibrated"),
        scope: Scope::Manuscript { scene_ids },
        evidence: Vec::new(),
        fingerprint: String::new(),
    };
    finding.fingerprint = super::comparison::fingerprint(
        &super::comparison::contextual_anchor(document_id, &finding),
        0,
    );
    finding
}

pub(crate) fn context(text: &str, start: usize, end: usize) -> EvidenceContext {
    let before = text[..start]
        .chars()
        .rev()
        .take_while(|ch| !matches!(ch, '.' | '!' | '?' | '\n' | '\r'))
        .take(64)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    let mut after = String::new();
    for ch in text[end..].chars().take(64) {
        after.push(ch);
        if matches!(ch, '.' | '!' | '?' | '\n' | '\r') {
            break;
        }
    }
    EvidenceContext { before, after }
}
