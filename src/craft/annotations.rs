use crate::{
    Error, Result,
    manuscript::{Evidence, Manuscript, author_source::validate_sources},
    structure::author_events::{AuthorSource, MAX_AUTHOR_EVENT_BYTES, MAX_AUTHOR_EVENTS},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthorAnnotationCoverage {
    pub scene_id: String,
    pub annotation_count: usize,
    pub covered_utf16: u32,
    pub total_utf16: u32,
}

pub(crate) struct AnnotationBudget {
    bytes: usize,
}
impl AnnotationBudget {
    pub fn new(manuscript: &Manuscript, events: usize) -> Result<Self> {
        manuscript.validate()?;
        if events > MAX_AUTHOR_EVENTS {
            return Err(invalid("annotation count exceeds 10000"));
        }
        Ok(Self { bytes: 0 })
    }
    pub fn add(&mut self, value: &str) -> Result<()> {
        self.bytes = self
            .bytes
            .checked_add(value.len())
            .filter(|bytes| *bytes <= MAX_AUTHOR_EVENT_BYTES)
            .ok_or_else(|| invalid("annotation metadata exceeds 4 MiB"))?;
        Ok(())
    }
    pub fn source(&mut self, source: &AuthorSource) -> Result<()> {
        self.add(&source.scene_id)?;
        self.add(&source.quote)
    }
}
pub(crate) fn invalid(message: &str) -> Error {
    Error::InvalidInput(message.into())
}
pub(crate) fn validate_text(value: &str) -> Result<()> {
    if value.trim().is_empty() || value.len() > 4096 {
        return Err(invalid("annotation text requires 1..=4096 bytes"));
    }
    Ok(())
}
pub(crate) fn validate(
    manuscript: &Manuscript,
    sources: &[&AuthorSource],
) -> Result<Vec<Evidence>> {
    validate_sources(manuscript, sources)
}
pub(crate) fn order(manuscript: &Manuscript) -> BTreeMap<&str, usize> {
    manuscript
        .scenes
        .iter()
        .enumerate()
        .map(|(i, scene)| (scene.id.as_str(), i))
        .collect()
}
pub(crate) fn character(manuscript: &Manuscript, name: &str) -> Result<String> {
    crate::manuscript::config::validate_name(name)?;
    if manuscript.known_characters.is_empty() {
        return Ok(name.into());
    }
    manuscript
        .known_characters
        .iter()
        .find(|candidate| candidate.to_lowercase() == name.to_lowercase())
        .cloned()
        .ok_or_else(|| invalid("annotation character is not declared in the manuscript cast"))
}
pub(crate) fn coverage(
    manuscript: &Manuscript,
    evidence: &[&Evidence],
) -> Vec<AuthorAnnotationCoverage> {
    let mut ranges = BTreeMap::<&str, Vec<(u32, u32)>>::new();
    for source in evidence {
        ranges
            .entry(&source.scene_id)
            .or_default()
            .push((source.range.utf16_start, source.range.utf16_end));
    }
    manuscript
        .scenes
        .iter()
        .map(|scene| {
            let mut spans = ranges.remove(scene.id.as_str()).unwrap_or_default();
            let annotation_count = spans.len();
            spans.sort_unstable();
            let mut covered = 0;
            let mut active = None::<(u32, u32)>;
            for (start, end) in spans {
                match active {
                    Some((a, b)) if start <= b => active = Some((a, b.max(end))),
                    Some((a, b)) => {
                        covered += b - a;
                        active = Some((start, end));
                    }
                    None => active = Some((start, end)),
                }
            }
            if let Some((a, b)) = active {
                covered += b - a;
            }
            AuthorAnnotationCoverage {
                scene_id: scene.id.clone(),
                annotation_count,
                covered_utf16: covered,
                total_utf16: u32::try_from(scene.text.encode_utf16().count())
                    .expect("validated scene size"),
            }
        })
        .collect()
}
