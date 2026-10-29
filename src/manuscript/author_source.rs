use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{Evidence, EvidenceRole, Manuscript, TextRange, config::validate_id, finding};
use crate::{Error, Result, substrate::offsets::OffsetIndex};

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct AuthorSource {
    pub scene_id: String,
    pub utf16_start: u32,
    pub utf16_end: u32,
    pub quote: String,
}

pub(crate) fn validate_sources(
    manuscript: &Manuscript,
    sources: &[&AuthorSource],
) -> Result<Vec<Evidence>> {
    let invalid = |message: &str| Error::InvalidInput(message.into());
    let scenes: BTreeMap<_, _> = manuscript
        .scenes
        .iter()
        .map(|scene| (scene.id.as_str(), scene.text.as_str()))
        .collect();
    let mut grouped: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (position, source) in sources.iter().enumerate() {
        validate_id(&source.scene_id)?;
        if source.utf16_start >= source.utf16_end
            || source.quote.trim().is_empty()
            || source.quote.len() > 64 * 1024
            || !scenes.contains_key(source.scene_id.as_str())
        {
            return Err(invalid(
                "author evidence requires a nonempty bounded quote in a known scene",
            ));
        }
        grouped.entry(&source.scene_id).or_default().push(position);
    }
    let mut evidence = vec![None; sources.len()];
    for (scene_id, positions) in grouped {
        let text = scenes[scene_id];
        let offsets = OffsetIndex::new(text)?;
        for position in positions {
            let source = sources[position];
            let (Some(start), Some(end)) = (
                offsets.byte(source.utf16_start),
                offsets.byte(source.utf16_end),
            ) else {
                return Err(invalid(
                    "author evidence range splits a surrogate pair or exceeds its scene",
                ));
            };
            if text.get(start..end) != Some(source.quote.as_str()) {
                return Err(invalid(
                    "author evidence quote does not match the original source range",
                ));
            }
            evidence[position] = Some(Evidence {
                document_id: manuscript.document_id.clone(),
                scene_id: scene_id.into(),
                range: TextRange {
                    byte_start: start,
                    byte_end: end,
                    utf16_start: source.utf16_start,
                    utf16_end: source.utf16_end,
                },
                quote: source.quote.clone(),
                role: EvidenceRole::Primary,
                context: Some(finding::context(text, start, end)),
            });
        }
    }
    Ok(evidence
        .into_iter()
        .map(|source| source.expect("all author sources validated"))
        .collect())
}
