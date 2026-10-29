//! Fast, local writing mechanics checks. This deliberately stays conservative:
//! it reports high-confidence mechanical issues and unknown lowercase words, while
//! leaving names, invented language, dialect, and stylistic choices alone.

use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingKind {
    Spelling,
    Repetition,
    Whitespace,
    Capitalization,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrammarFinding {
    pub kind: FindingKind,
    pub utf16_start: u32,
    pub utf16_end: u32,
    pub message: String,
    #[serde(default)]
    pub replacements: Vec<String>,
}

pub fn check(text: &str) -> Vec<GrammarFinding> {
    check_bounded(text, usize::MAX).0
}

pub(crate) fn check_bounded(text: &str, limit: usize) -> (Vec<GrammarFinding>, bool) {
    check_selected(text, limit, &Selection::default())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Selection {
    pub repetition: bool,
    pub spelling: bool,
    pub whitespace: bool,
    pub dictionary: std::sync::Arc<std::collections::BTreeSet<String>>,
}
impl Default for Selection {
    fn default() -> Self {
        Self {
            repetition: true,
            spelling: true,
            whitespace: true,
            dictionary: Default::default(),
        }
    }
}
pub(crate) fn check_selected(
    text: &str,
    limit: usize,
    selection: &Selection,
) -> (Vec<GrammarFinding>, bool) {
    let mut findings = Vec::new();
    if !(selection.repetition || selection.spelling || selection.whitespace) {
        return (findings, false);
    }
    let offsets = crate::substrate::offsets::OffsetIndex::new(text)
        .expect("grammar text fits offset coordinates");
    let words = word_spans(text);

    for pair in words.windows(2) {
        if findings.len() > limit {
            break;
        }
        let (a, b) = (&pair[0], &pair[1]);
        if selection.repetition
            && a.word.eq_ignore_ascii_case(b.word)
            && separator_is_only_space(text, a.end, b.start)
        {
            findings.push(finding(
                FindingKind::Repetition,
                &offsets,
                b.start,
                b.end,
                format!("Repeated word ‘{}’.", b.word),
                vec![],
            ));
        }
    }

    for span in &words {
        if findings.len() > limit {
            break;
        }
        let lower = span.word.to_lowercase();
        let looks_like_name = span.word.chars().next().is_some_and(char::is_uppercase);
        let possessive_or_contraction = span.word.contains(['\'', '’']);
        if selection.spelling
            && !selection.dictionary.contains(&lower)
            && span.word.chars().count() >= 3
            && !looks_like_name
            && !possessive_or_contraction
            && crate::substrate::wordfreq::zipf_frequency(&lower) == 0.0
        {
            findings.push(finding(
                FindingKind::Spelling,
                &offsets,
                span.start,
                span.end,
                format!("‘{}’ is not in the local dictionary.", span.word),
                crate::substrate::wordfreq::spelling_suggestions(&lower, 3),
            ));
        }
    }

    let bytes = text.as_bytes();
    let mut index = 0;
    while index + 1 < bytes.len() && findings.len() <= limit {
        if selection.whitespace && bytes[index] == b' ' && bytes[index + 1] == b' ' {
            let start = index + 1;
            let mut end = start;
            while end < bytes.len() && bytes[end] == b' ' {
                end += 1;
            }
            findings.push(finding(
                FindingKind::Whitespace,
                &offsets,
                start,
                end,
                "Multiple spaces.".into(),
                vec![String::new()],
            ));
            index = end;
        } else {
            index += 1;
        }
    }

    let limited = findings.len() > limit;
    findings.truncate(limit);
    findings.sort_by_key(|item| (item.utf16_start, item.utf16_end));
    (findings, limited)
}

struct WordSpan<'a> {
    word: &'a str,
    start: usize,
    end: usize,
}

fn word_spans(text: &str) -> Vec<WordSpan<'_>> {
    let mut spans = Vec::new();
    let mut start = None;
    for (index, ch) in text.char_indices() {
        let word_char = ch.is_alphabetic() || (start.is_some() && matches!(ch, '\'' | '’'));
        match (start, word_char) {
            (None, true) => start = Some(index),
            (Some(begin), false) => {
                spans.push(WordSpan {
                    word: &text[begin..index],
                    start: begin,
                    end: index,
                });
                start = None;
            }
            _ => {}
        }
    }
    if let Some(begin) = start {
        spans.push(WordSpan {
            word: &text[begin..],
            start: begin,
            end: text.len(),
        });
    }
    spans
}

fn separator_is_only_space(text: &str, start: usize, end: usize) -> bool {
    let between = &text[start..end];
    !between.is_empty()
        && between.chars().all(char::is_whitespace)
        && !between.contains(['.', '!', '?', ',', ';', ':'])
}

fn finding(
    kind: FindingKind,
    offsets: &crate::substrate::offsets::OffsetIndex<'_>,
    start: usize,
    end: usize,
    message: String,
    replacements: Vec<String>,
) -> GrammarFinding {
    GrammarFinding {
        kind,
        utf16_start: offsets
            .utf16(start)
            .expect("word boundary belongs to original text"),
        utf16_end: offsets
            .utf16(end)
            .expect("word boundary belongs to original text"),
        message,
        replacements,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catches_repetition_spacing_and_misspelling_without_flagging_names() {
        let found = check("Mara saw saw a  qwick fox.");
        assert!(found.iter().any(|f| f.kind == FindingKind::Repetition));
        assert!(found.iter().any(|f| f.kind == FindingKind::Whitespace));
        assert!(
            found.iter().any(
                |f| f.kind == FindingKind::Spelling && f.replacements.contains(&"quick".into())
            )
        );
        assert!(!found.iter().any(|f| f.message.contains("Mara")));
    }

    #[test]
    fn offsets_are_utf16() {
        let found = check("😀 go go");
        let repeated = found
            .iter()
            .find(|f| f.kind == FindingKind::Repetition)
            .unwrap();
        assert_eq!(repeated.utf16_start, 6);
    }
}
