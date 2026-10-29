use super::{Coverage, Finding, ModelPolicy, Report, Scope, Status};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchKind {
    Context,
    Evidence,
    AmbiguousOccurrence,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PersistentFinding {
    pub before: Finding,
    pub after: Finding,
    pub match_kind: MatchKind,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RevisionComparison {
    #[serde(default)]
    pub complete: bool,
    #[serde(default)]
    pub unassessed: Vec<UnassessedFinding>,
    pub introduced: Vec<Finding>,
    pub persistent: Vec<PersistentFinding>,
    pub resolved: Vec<Finding>,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeDirection {
    Introduced,
    Resolved,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UnassessedFinding {
    pub finding: Finding,
    pub direction: ChangeDirection,
    pub reason: String,
}

impl Coverage {
    pub fn is_intentionally_disabled(&self) -> bool {
        self.status == Status::Unavailable
            && self.diagnostics.iter().any(|reason| {
                matches!(
                    reason.as_str(),
                    "disabled"
                        | "analysis_disabled"
                        | "model_analysis_disabled"
                        | "rule_not_selected"
                        | "rules_not_selected"
                        | "lexicon_empty"
                        | "rule_not_selected_or_lexicon_empty"
                        | "model_analysis_disabled_or_rule_not_selected"
                ) || reason.starts_with("author_intends")
            })
    }
}

fn eligible_scene_units(report: &Report, analyzer: &str) -> Option<usize> {
    if report.metrics.len() != report.scene_ids.len() {
        return None;
    }
    let configuration = report.configuration.as_ref()?;
    match analyzer {
        "readability" | "pacing" => Some(
            report
                .metrics
                .iter()
                .filter(|scene| scene.word_count >= configuration.min_scene_words())
                .count(),
        ),
        "grammar" | "prose" | "continuity" | "structure" => Some(
            report
                .metrics
                .iter()
                .filter(|scene| scene.word_count > 0 || scene.sentence_count > 0)
                .count(),
        ),
        _ => None,
    }
}

fn has_no_eligible_work(report: &Report, coverage: &Coverage) -> bool {
    match coverage.analyzer.as_str() {
        "nli" => coverage.total_units == 0,
        "structure" => eligible_scene_units(report, "structure").is_some_and(|count| {
            count
                < report
                    .configuration
                    .as_ref()
                    .expect("configuration required for eligibility")
                    .thresholds
                    .min_structure_scenes
                    .unwrap_or(3)
        }),
        analyzer => eligible_scene_units(report, analyzer) == Some(0),
    }
}

fn coverage_is_assessed(report: &Report, coverage: &Coverage) -> bool {
    if coverage.is_intentionally_disabled() {
        return true;
    }
    match coverage.status {
        Status::Complete => true,
        Status::Unavailable | Status::ExecutionFailure => false,
        Status::InsufficientInput => {
            coverage.analyzed_units == 0 && has_no_eligible_work(report, coverage)
        }
        Status::HeuristicFallback => {
            if coverage.analyzer == "continuity"
                && report.diagnostics.iter().any(|d| {
                    matches!(
                        d.code.as_str(),
                        "chronology_ambiguous" | "ambiguous_constraint"
                    )
                })
            {
                return false;
            }
            coverage.analyzed_units == coverage.total_units
                || eligible_scene_units(report, &coverage.analyzer) == Some(coverage.analyzed_units)
        }
    }
}

fn selected_coverage_is_assessed(report: &Report) -> bool {
    let Some(configuration) = &report.configuration else {
        return false;
    };
    let required: BTreeSet<_> = super::rule_definitions()
        .into_iter()
        .filter(|rule| {
            rule.facade_available
                && configuration.selected(rule.id)
                && rule.default_severity >= configuration.minimum_severity
                && (!rule.requires_model || configuration.nli != ModelPolicy::Disabled)
                && (rule.analyzer != "prose" || !configuration.lexicon.is_empty())
        })
        .map(|rule| rule.analyzer)
        .collect();
    required.into_iter().all(|analyzer| {
        let mut entries = report
            .coverage
            .iter()
            .filter(|entry| entry.analyzer == analyzer);
        let Some(coverage) = entries.next() else {
            return false;
        };
        entries.next().is_none() && coverage_is_assessed(report, coverage)
    })
}

impl RevisionComparison {
    pub fn is_complete(&self, before: &Report, after: &Report) -> bool {
        before.document_id == after.document_id
            && self.unassessed.is_empty()
            && [before, after].iter().all(|report| {
                report.schema_version == "2"
                    && report.validate().is_ok()
                    && selected_coverage_is_assessed(report)
                    && report.summary.omitted_findings == 0
                    && report.summary.omitted_suppressed == 0
                    && !report
                        .diagnostics
                        .iter()
                        .any(|d| d.code.ends_with("candidate_limit") || d.code == "finding_limit")
                    && !report
                        .coverage
                        .iter()
                        .any(|c| c.diagnostics.iter().any(|d| d.contains("candidate_limit")))
            })
    }
}

fn field(key: &mut String, value: &str) {
    key.push_str(&format!("{}:{value}", value.len()));
}
fn normalized(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}
pub(crate) fn anchor(document: &str, f: &Finding) -> String {
    let mut key = String::from("nl-anchor-v1:");
    field(&mut key, document);
    field(&mut key, &f.rule_id);
    match &f.scope {
        Scope::Passage { scene_id } | Scope::Scene { scene_id } => {
            field(&mut key, "scene");
            field(&mut key, scene_id);
        }
        Scope::Manuscript { scene_ids } => {
            field(&mut key, "manuscript");
            let mut ids = scene_ids.clone();
            ids.sort();
            for id in ids {
                field(&mut key, &id);
            }
            field(&mut key, &f.message);
        }
    }
    for e in f
        .evidence
        .iter()
        .filter(|e| e.role != super::EvidenceRole::Context)
    {
        field(&mut key, &e.scene_id);
        field(&mut key, &normalized(&e.quote));
    }
    key
}
pub(crate) fn contextual_anchor(document: &str, f: &Finding) -> String {
    let mut key = anchor(document, f);
    for e in f
        .evidence
        .iter()
        .filter(|e| e.role != super::EvidenceRole::Context)
    {
        if let Some(c) = &e.context {
            field(&mut key, &normalized(&c.before));
            field(&mut key, &normalized(&c.after));
        } else {
            field(&mut key, "");
            field(&mut key, "");
        }
    }
    key
}
pub(crate) fn fingerprint(key: &str, occurrence: usize) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for b in key.bytes().chain((occurrence as u64).to_le_bytes()) {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("nl-v2-{hash:016x}-{occurrence}")
}
/// Match exact local contexts first, then unique evidence, then explicitly
/// ambiguous duplicate occurrences. Full anchors protect against hash collisions.
pub fn compare_reports(before: &Report, after: &Report) -> RevisionComparison {
    let before_keys = before
        .findings
        .iter()
        .map(|f| anchor(&before.document_id, f))
        .collect::<Vec<_>>();
    let after_keys = after
        .findings
        .iter()
        .map(|f| anchor(&after.document_id, f))
        .collect::<Vec<_>>();
    let after_contexts = after
        .findings
        .iter()
        .map(|f| contextual_anchor(&after.document_id, f))
        .collect::<Vec<_>>();
    let mut remaining = BTreeMap::<&str, BTreeSet<usize>>::new();
    let mut contexts = BTreeMap::<String, (usize, VecDeque<usize>)>::new();
    let mut after_counts = BTreeMap::<&str, usize>::new();
    for (i, (f, key)) in before.findings.iter().zip(&before_keys).enumerate() {
        remaining.entry(key.as_str()).or_default().insert(i);
        let group = contexts
            .entry(contextual_anchor(&before.document_id, f))
            .or_default();
        group.0 += 1;
        group.1.push_back(i);
    }
    for context in &after_contexts {
        *after_counts.entry(context).or_default() += 1;
    }
    let mut used = vec![false; before.findings.len()];
    let mut matched = vec![None; after.findings.len()];
    for (i, context) in after_contexts.iter().enumerate() {
        if let Some((count, group)) = contexts.get_mut(context)
            && let Some(j) = group.pop_front()
        {
            used[j] = true;
            remaining
                .get_mut(before_keys[j].as_str())
                .expect("matching group")
                .remove(&j);
            matched[i] = Some((
                j,
                if *count > 1 || after_counts[context.as_str()] > 1 {
                    MatchKind::AmbiguousOccurrence
                } else {
                    MatchKind::Context
                },
            ));
        }
    }
    let mut unmatched_counts = BTreeMap::<&str, usize>::new();
    for (i, key) in after_keys.iter().enumerate() {
        if matched[i].is_none() {
            *unmatched_counts.entry(key).or_default() += 1;
        }
    }
    for (i, key) in after_keys.iter().enumerate() {
        if matched[i].is_some() {
            continue;
        }
        if let Some(group) = remaining.get_mut(key.as_str()) {
            let ambiguous = group.len() > 1 || unmatched_counts[key.as_str()] > 1;
            if let Some(j) = group.pop_first() {
                used[j] = true;
                matched[i] = Some((
                    j,
                    if ambiguous {
                        MatchKind::AmbiguousOccurrence
                    } else {
                        MatchKind::Evidence
                    },
                ));
            }
        }
    }
    let mut diff = RevisionComparison::default();
    for (i, f) in after.findings.iter().enumerate() {
        if let Some((j, kind)) = matched[i] {
            diff.persistent.push(PersistentFinding {
                before: before.findings[j].clone(),
                after: f.clone(),
                match_kind: kind,
            });
        } else if let Some(reason) = assessment_gap(before, f) {
            diff.unassessed.push(UnassessedFinding {
                finding: f.clone(),
                direction: ChangeDirection::Introduced,
                reason,
            });
        } else {
            diff.introduced.push(f.clone());
        }
    }
    for (i, f) in before.findings.iter().enumerate() {
        if !used[i] {
            if let Some(reason) = assessment_gap(after, f) {
                diff.unassessed.push(UnassessedFinding {
                    finding: f.clone(),
                    direction: ChangeDirection::Resolved,
                    reason,
                });
            } else {
                diff.resolved.push(f.clone());
            }
        }
    }
    diff.complete = diff.is_complete(before, after);
    diff
}
fn assessment_gap(report: &Report, finding: &Finding) -> Option<String> {
    let scene = match &finding.scope {
        Scope::Scene { scene_id } | Scope::Passage { scene_id } => Some(scene_id),
        Scope::Manuscript { .. } => None,
    };
    if scene.is_some_and(|id| !report.scene_ids.contains(id)) {
        return None;
    }
    if let Some(config) = &report.configuration {
        if !config.selected(&finding.rule_id) || finding.severity < config.minimum_severity {
            return Some("rule_not_evaluated".into());
        }
        if matches!(finding.analyzer.as_str(), "readability" | "pacing")
            && scene
                .and_then(|id| report.metrics.iter().find(|m| &m.scene_id == id))
                .is_some_and(|m| m.word_count < config.min_scene_words())
        {
            return Some("insufficient_scene_input".into());
        }
    }
    if report.summary.omitted_findings > 0 {
        return Some("report_finding_limit".into());
    }
    let analyzer = if finding.rule_id == "NL-CONTINUITY-NLI" {
        "nli"
    } else {
        finding.analyzer.as_str()
    };
    let Some(coverage) = report.coverage.iter().find(|c| c.analyzer == analyzer) else {
        return Some("coverage_missing".into());
    };
    match coverage.status {
        super::Status::Unavailable => Some("capability_unavailable".into()),
        super::Status::ExecutionFailure => Some("capability_failed".into()),
        super::Status::InsufficientInput => Some("insufficient_input".into()),
        super::Status::HeuristicFallback => {
            let limited = report
                .diagnostics
                .iter()
                .any(|d| d.code.ends_with("candidate_limit") || d.code == "chronology_ambiguous");
            if analyzer == "nli"
                || limited
                    && report.diagnostics.iter().any(|d| {
                        d.scene_ids.is_empty() || scene.is_none_or(|id| d.scene_ids.contains(id))
                    })
            {
                Some("partial_analysis".into())
            } else {
                None
            }
        }
        super::Status::Complete => None,
    }
}

pub fn try_compare_reports(before: &Report, after: &Report) -> crate::Result<RevisionComparison> {
    before.validate()?;
    after.validate()?;
    if before.document_id != after.document_id {
        return Err(crate::Error::InvalidInput(
            "revision documents have different identities".into(),
        ));
    }
    Ok(compare_reports(before, after))
}

pub fn assign_fingerprints(document: &str, findings: &mut [Finding]) -> crate::Result<()> {
    super::config::validate_id(document)?;
    if findings.iter().any(|f| f.document_id != document) {
        return Err(crate::Error::InvalidInput(
            "finding document identity differs".into(),
        ));
    }
    let mut ordinals = BTreeMap::new();
    for f in findings {
        let key = contextual_anchor(document, f);
        let n = ordinals.entry(key.clone()).or_insert(0);
        f.fingerprint = fingerprint(&key, *n);
        *n += 1;
    }
    Ok(())
}
