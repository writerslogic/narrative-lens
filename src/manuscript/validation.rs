use super::*;
use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};
fn invalid(message: &str) -> Error {
    Error::InvalidInput(message.into())
}
impl Report {
    pub fn validate(&self) -> Result<()> {
        if !matches!(self.schema_version.as_str(), "1" | "2") {
            return Err(invalid("unsupported report schema version"));
        }
        config::validate_id(&self.document_id)?;
        if let Some(configuration) = &self.configuration {
            configuration.validate()?;
        }
        if self.model_identities.len() > 1 {
            return Err(invalid(
                "report contains duplicate or unsupported model identities",
            ));
        }
        for identity in &self.model_identities {
            identity
                .validate(crate::onnx::checked::ModelKind::Nli)
                .map_err(|_| invalid("report contains invalid model identity metadata"))?;
            if self.configuration.as_ref().is_some_and(|configuration| {
                configuration.nli == ModelPolicy::Disabled
                    || !configuration.selected("NL-CONTINUITY-NLI")
            }) {
                return Err(invalid(
                    "model identity conflicts with disabled NLI configuration",
                ));
            }
            if !self.coverage.iter().any(|coverage| {
                coverage.analyzer == "nli"
                    && coverage.analyzed_units > 0
                    && matches!(
                        coverage.status,
                        Status::Complete | Status::HeuristicFallback
                    )
            }) {
                return Err(invalid(
                    "model identity requires successfully analyzed NLI coverage",
                ));
            }
        }
        if self.scene_ids.len() > 10_000
            || self.findings.len() > 100_000
            || self.suppressed.len() > 100_000
        {
            return Err(invalid("report exceeds collection limits"));
        }
        let ids = self.scene_ids.iter().collect::<BTreeSet<_>>();
        if ids.len() != self.scene_ids.len() {
            return Err(invalid("duplicate report scene IDs"));
        }
        for id in &ids {
            config::validate_id(id)?;
        }
        if self.coverage.len() > 128
            || self.diagnostics.len() > 100_000
            || self.chronology.reader_knowledge.len() > ids.len()
            || self
                .chronology
                .ambiguous_scene_ids
                .iter()
                .any(|id| !ids.contains(id))
        {
            return Err(invalid("invalid report coverage or chronology"));
        }
        let mut ordered = BTreeSet::new();
        for (lane, scenes) in &self.chronology.lanes {
            config::validate_id(lane)?;
            if scenes
                .iter()
                .any(|id| !ids.contains(id) || !ordered.insert(id))
            {
                return Err(invalid("chronology refers to an unknown or repeated scene"));
            }
        }
        let mut previous = None;
        for reader in &self.chronology.reader_knowledge {
            let position = self
                .scene_ids
                .iter()
                .position(|id| id == &reader.after_scene_id)
                .ok_or_else(|| invalid("reader knowledge refers to an unknown scene"))?;
            if previous.is_some_and(|p| p >= position)
                || reader
                    .introduced
                    .iter()
                    .any(|e| e.scene_id != reader.after_scene_id)
            {
                return Err(invalid("reader knowledge must follow presentation order"));
            }
            previous = Some(position);
            for e in &reader.introduced {
                validate_evidence(e, &self.document_id, &ids)?;
            }
        }
        if self
            .diagnostics
            .iter()
            .any(|d| d.scene_ids.iter().any(|id| !ids.contains(id)))
        {
            return Err(invalid("diagnostic refers to an unknown scene"));
        }
        let mut fingerprints = BTreeSet::new();
        for f in self
            .findings
            .iter()
            .chain(self.suppressed.iter().map(|s| &s.finding))
        {
            if f.document_id != self.document_id
                || f.rule_id.is_empty()
                || f.rule_id.len() > 128
                || !f
                    .rule_id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
                || f.analyzer.trim().is_empty()
                || f.analyzer.len() > 128
            {
                return Err(invalid("invalid finding identity"));
            }
            if !f.confidence.score.is_finite()
                || !(0.0..=1.0).contains(&f.confidence.score)
                || f.confidence.method.trim().is_empty()
                || f.confidence.method.len() > 1024
            {
                return Err(invalid("invalid confidence provenance"));
            }
            if f.confidence.source == ConfidenceSource::Calibrated
                && f.confidence
                    .calibration_id
                    .as_ref()
                    .is_none_or(|s| s.trim().is_empty())
            {
                return Err(invalid(
                    "calibrated confidence requires calibration identity",
                ));
            }
            if f.fingerprint.is_empty()
                || f.fingerprint.len() > 128
                || !fingerprints.insert(&f.fingerprint)
            {
                return Err(invalid("missing or duplicate finding fingerprint"));
            }
            match &f.scope {
                Scope::Scene { scene_id } | Scope::Passage { scene_id } => {
                    if !ids.contains(scene_id) {
                        return Err(invalid("finding refers to unknown scene"));
                    }
                }
                Scope::Manuscript { scene_ids } => {
                    if scene_ids.iter().any(|id| !ids.contains(id)) {
                        return Err(invalid("aggregate scope refers to unknown scene"));
                    }
                }
            }
            if matches!(f.scope, Scope::Passage { .. })
                && !f.evidence.iter().any(|e| e.role == EvidenceRole::Primary)
            {
                return Err(invalid("passage finding needs primary evidence"));
            }
            for e in &f.evidence {
                validate_evidence(e, &self.document_id, &ids)?;
                if e.role == EvidenceRole::Primary
                    && matches!(&f.scope,Scope::Scene{scene_id}|Scope::Passage{scene_id} if scene_id!=&e.scene_id)
                {
                    return Err(invalid("primary evidence disagrees with finding scope"));
                }
            }
        }
        for c in &self.coverage {
            if c.analyzed_units > c.total_units {
                return Err(invalid("coverage exceeds input units"));
            }
        }
        let metric_ids = self
            .metrics
            .iter()
            .map(|m| &m.scene_id)
            .collect::<BTreeSet<_>>();
        if metric_ids.len() != self.metrics.len()
            || metric_ids.iter().any(|id| !ids.contains(*id))
            || self.metrics.iter().any(|m| {
                [m.grade, m.sentence_words, m.dialogue_ratio, m.tension]
                    .iter()
                    .any(|v| !v.is_finite())
            })
        {
            return Err(invalid("invalid or duplicate scene metrics"));
        }
        if self.schema_version == "2" {
            let s = &self.summary;
            if s.returned_findings != self.findings.len()
                || s.suppressed_findings.checked_sub(s.omitted_suppressed)
                    != Some(self.suppressed.len())
                || s.total_findings
                    .checked_sub(s.suppressed_findings)
                    .and_then(|n| n.checked_sub(s.omitted_findings))
                    != Some(self.findings.len())
                || s.severity_counts
                    .values()
                    .try_fold(0usize, |n, v| n.checked_add(*v))
                    != s.total_findings.checked_sub(s.suppressed_findings)
            {
                return Err(invalid("report summary does not match finding counts"));
            }
        }
        Ok(())
    }
    pub fn validate_against(&self, manuscript: &Manuscript) -> Result<()> {
        self.validate()?;
        manuscript.validate()?;
        if self.document_id != manuscript.document_id
            || self.scene_ids
                != manuscript
                    .scenes
                    .iter()
                    .map(|s| s.id.clone())
                    .collect::<Vec<_>>()
        {
            return Err(invalid("report and manuscript identities differ"));
        }
        let scenes: BTreeMap<_, _> = manuscript
            .scenes
            .iter()
            .map(|s| (s.id.as_str(), s.text.as_str()))
            .collect();
        let mut indices = BTreeMap::new();
        for e in self
            .findings
            .iter()
            .chain(self.suppressed.iter().map(|s| &s.finding))
            .flat_map(|f| &f.evidence)
            .chain(
                self.chronology
                    .reader_knowledge
                    .iter()
                    .flat_map(|r| &r.introduced),
            )
        {
            let text = scenes[e.scene_id.as_str()];
            if !indices.contains_key(e.scene_id.as_str()) {
                indices.insert(
                    e.scene_id.as_str(),
                    crate::substrate::offsets::OffsetIndex::new(text)?,
                );
            }
            let index = &indices[e.scene_id.as_str()];
            if index.utf16(e.range.byte_start) != Some(e.range.utf16_start)
                || index.utf16(e.range.byte_end) != Some(e.range.utf16_end)
                || text.get(e.range.byte_start..e.range.byte_end) != Some(e.quote.as_str())
            {
                return Err(invalid("evidence does not match original manuscript text"));
            }
            if e.context
                .as_ref()
                .is_some_and(|c| c != &finding::context(text, e.range.byte_start, e.range.byte_end))
            {
                return Err(invalid(
                    "evidence context does not match original manuscript text",
                ));
            }
        }
        Ok(())
    }
}

fn validate_evidence(e: &Evidence, document: &str, ids: &BTreeSet<&String>) -> Result<()> {
    if e.document_id != document || !ids.contains(&e.scene_id) {
        return Err(invalid(
            "evidence refers to a foreign document or unknown scene",
        ));
    }
    if e.range.byte_start > e.range.byte_end
        || e.range.utf16_start > e.range.utf16_end
        || e.range.byte_end > 16 * 1024 * 1024
        || e.range.utf16_start as usize > e.range.byte_start
        || e.range.utf16_end as usize > e.range.byte_end
        || e.range.byte_end - e.range.byte_start != e.quote.len()
        || usize::try_from(e.range.utf16_end - e.range.utf16_start).ok()
            != Some(e.quote.encode_utf16().count())
    {
        return Err(invalid("evidence range and quote lengths disagree"));
    }
    if let Some(c) = &e.context
        && (c.before.chars().count() > 64 || c.after.chars().count() > 64)
    {
        return Err(invalid("evidence context exceeds 64 characters"));
    }
    Ok(())
}
