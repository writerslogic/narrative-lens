//! Shared original-text coordinate conversions for a batch of findings.
use crate::{Error, Result};

pub struct OffsetIndex<'a> {
    text: &'a str,
    boundaries: Vec<(u32, u32)>,
}
impl<'a> OffsetIndex<'a> {
    pub fn new(text: &'a str) -> Result<Self> {
        let byte_len = u32::try_from(text.len())
            .map_err(|_| Error::InvalidInput("text exceeds offset index limit".into()))?;
        let mut boundaries = Vec::with_capacity(text.chars().count() + 1);
        let mut units = 0u32;
        for (byte, ch) in text.char_indices() {
            boundaries.push((byte as u32, units));
            units = units
                .checked_add(ch.len_utf16() as u32)
                .ok_or_else(|| Error::InvalidInput("text exceeds UTF-16 limit".into()))?;
        }
        boundaries.push((byte_len, units));
        Ok(Self { text, boundaries })
    }
    pub fn text(&self) -> &'a str {
        self.text
    }
    pub fn utf16(&self, byte: usize) -> Option<u32> {
        let byte = u32::try_from(byte).ok()?;
        self.boundaries
            .binary_search_by_key(&byte, |p| p.0)
            .ok()
            .map(|i| self.boundaries[i].1)
    }
    pub fn byte(&self, utf16: u32) -> Option<usize> {
        self.boundaries
            .binary_search_by_key(&utf16, |p| p.1)
            .ok()
            .map(|i| self.boundaries[i].0 as usize)
    }
}
