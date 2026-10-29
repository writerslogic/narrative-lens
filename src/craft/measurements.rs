use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const MAX_TEXT_BYTES: usize = 16 * 1024 * 1024;
pub(crate) const MAX_TOTAL_BYTES: usize = 64 * 1024 * 1024;
pub(crate) const MAX_TURNS: usize = 10_000;

pub(crate) fn invalid(message: &str) -> Error {
    Error::InvalidInput(message.into())
}

pub(crate) fn validate_texts<'a>(texts: impl IntoIterator<Item = &'a str>) -> Result<()> {
    let mut count = 0usize;
    let mut bytes = 0usize;
    for text in texts {
        count += 1;
        bytes = bytes
            .checked_add(text.len())
            .ok_or_else(|| invalid("dialogue byte budget exceeded"))?;
        if count > MAX_TURNS || text.len() > MAX_TEXT_BYTES || bytes > MAX_TOTAL_BYTES {
            return Err(invalid(
                "dialogue exceeds 10000 turns, 16 MiB per turn or 64 MiB total",
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<()> {
    let mut seen = BTreeSet::new();
    for name in names {
        crate::manuscript::config::validate_name(name)?;
        if !seen.insert(name.to_lowercase()) {
            return Err(invalid("character names must be unique ignoring case"));
        }
        if seen.len() > 256 {
            return Err(invalid("dialogue cast exceeds 256 characters"));
        }
    }
    Ok(())
}

pub(crate) fn tokens(text: &str) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    if text.contains(['’', '‘']) {
        crate::substrate::text::tokenize_words(&text.replace(['’', '‘'], "'"))
    } else {
        crate::substrate::text::tokenize_words(text)
    }
}

pub(crate) fn ranges(
    text: &str,
    spans: &[(usize, usize)],
) -> Result<Vec<crate::manuscript::TextRange>> {
    let mut points = BTreeMap::new();
    for &(start, end) in spans {
        if start > end
            || end > text.len()
            || !text.is_char_boundary(start)
            || !text.is_char_boundary(end)
        {
            return Err(invalid("cue range must follow original UTF-8 boundaries"));
        }
        points.insert(start, 0u32);
        points.insert(end, 0u32);
    }
    let mut previous = 0;
    let mut utf16 = 0u32;
    for (byte, units) in &mut points {
        utf16 = utf16
            .checked_add(text[previous..*byte].encode_utf16().count() as u32)
            .ok_or_else(|| invalid("cue text exceeds UTF-16 offset limit"))?;
        *units = utf16;
        previous = *byte;
    }
    Ok(spans
        .iter()
        .map(|&(start, end)| crate::manuscript::TextRange {
            byte_start: start,
            byte_end: end,
            utf16_start: points[&start],
            utf16_end: points[&end],
        })
        .collect())
}
