use super::*;
use std::collections::BTreeMap;
type Order = (usize, usize, String, String, usize);

pub(crate) struct Collector<'a> {
    manuscript: &'a Manuscript,
    order: BTreeMap<&'a str, usize>,
    active: BTreeMap<Order, Finding>,
    suppressed: BTreeMap<Order, SuppressedFinding>,
    bucket: String,
    ordinals: BTreeMap<String, usize>,
    sequence: usize,
    summary: ReportSummary,
}
impl<'a> Collector<'a> {
    pub fn new(manuscript: &'a Manuscript) -> Self {
        Self {
            manuscript,
            order: manuscript
                .scenes
                .iter()
                .enumerate()
                .map(|(i, s)| (s.id.as_str(), i))
                .collect(),
            active: Default::default(),
            suppressed: Default::default(),
            bucket: String::new(),
            ordinals: Default::default(),
            sequence: 0,
            summary: Default::default(),
        }
    }
    pub fn emit(&mut self, mut finding: Finding) {
        let config = &self.manuscript.configuration;
        if !config.selected(&finding.rule_id) || finding.severity < config.minimum_severity {
            return;
        }
        let scene = match &finding.scope {
            Scope::Scene { scene_id } | Scope::Passage { scene_id } => scene_id.as_str(),
            Scope::Manuscript { .. } => "",
        };
        if scene != self.bucket {
            self.ordinals.clear();
            self.bucket = scene.into();
        }
        let anchor = super::comparison::contextual_anchor(&self.manuscript.document_id, &finding);
        let ordinal = self.ordinals.entry(anchor.clone()).or_default();
        finding.fingerprint = super::comparison::fingerprint(&anchor, *ordinal);
        *ordinal += 1;
        let index = self.order.get(scene).copied().unwrap_or(usize::MAX);
        let start = finding.evidence.first().map_or(0, |e| e.range.byte_start);
        let key = (
            index,
            start,
            finding.rule_id.clone(),
            finding.message.clone(),
            self.sequence,
        );
        self.sequence += 1;
        self.summary.total_findings += 1;
        let matches_scene = |id: &str| match &finding.scope {
            Scope::Scene { scene_id } | Scope::Passage { scene_id } => scene_id == id,
            Scope::Manuscript { scene_ids } => scene_ids.iter().any(|s| s == id),
        };
        if let Some(s) = config.suppressions.iter().find(|s| {
            s.rule_id.as_ref().is_none_or(|r| r == &finding.rule_id)
                && s.scene_id.as_ref().is_none_or(|id| matches_scene(id))
                && s.fingerprint
                    .as_ref()
                    .is_none_or(|p| p == &finding.fingerprint)
        }) {
            self.summary.suppressed_findings += 1;
            self.suppressed.insert(
                key,
                SuppressedFinding {
                    finding,
                    reason: s.reason.clone(),
                },
            );
            if self.suppressed.len() > config.limits.max_findings {
                self.suppressed.pop_last();
            }
        } else {
            *self
                .summary
                .severity_counts
                .entry(finding.severity)
                .or_default() += 1;
            self.active.insert(key, finding);
            if self.active.len() > config.limits.max_findings {
                self.active.pop_last();
            }
        }
    }
    pub fn finish(mut self) -> (Vec<Finding>, Vec<SuppressedFinding>, ReportSummary) {
        self.summary.returned_findings = self.active.len();
        self.summary.omitted_findings =
            self.summary.total_findings - self.summary.suppressed_findings - self.active.len();
        self.summary.omitted_suppressed = self.summary.suppressed_findings - self.suppressed.len();
        (
            self.active.into_values().collect(),
            self.suppressed.into_values().collect(),
            self.summary,
        )
    }
}
