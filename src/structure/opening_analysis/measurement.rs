use super::*;
use crate::craft::{
    dialogue::{DialogueExtractionOptions, extract_dialogue_checked},
    measurements,
};
use crate::manuscript::TextRange;
use std::sync::OnceLock;

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpeningCueKind {
    ActionWord,
    QuestionMarker,
    FirstPersonPronoun,
    SettingPhrase,
    SensoryWord,
    BackstoryPhrase,
    CapitalizedToken,
    TopicWord,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpeningCue {
    pub kind: OpeningCueKind,
    pub range: TextRange,
    pub quote: String,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpeningParagraph {
    pub range: TextRange,
    pub word_count: usize,
    pub sentence_count: usize,
    pub backstory_sentence_count: usize,
    /// Paired quotation candidates whose opening delimiter is in this paragraph.
    pub quotation_candidate_count: usize,
}

pub(super) struct MeasuredOpening {
    pub first_line_range: Option<TextRange>,
    pub first_sentence_range: Option<TextRange>,
    pub paragraphs: Vec<OpeningParagraph>,
    pub cues: Vec<OpeningCue>,
    pub diagnostics: Vec<String>,
}

fn source_span(text: &str, part: &str) -> (usize, usize) {
    let start = part.as_ptr() as usize - text.as_ptr() as usize;
    (start, start + part.len())
}

pub(super) fn measure(text: &str) -> crate::Result<MeasuredOpening> {
    if text.len() > measurements::MAX_TEXT_BYTES {
        return Err(measurements::invalid("opening exceeds 16 MiB"));
    }
    let paragraphs = crate::substrate::text::split_paragraphs(text);
    if paragraphs.len() > 10_000 {
        return Err(measurements::invalid("opening exceeds 10000 paragraphs"));
    }
    let first = text.lines().next().unwrap_or("").trim();
    let first_line = (!first.is_empty()).then(|| source_span(text, first));
    let first_sentence = crate::substrate::utils::sentence_spans(text)
        .first()
        .copied();
    let mut found = Vec::new();
    for (kind, matcher) in [
        (OpeningCueKind::ActionWord, action_matcher()),
        (OpeningCueKind::QuestionMarker, question_matcher()),
        (OpeningCueKind::FirstPersonPronoun, voice_matcher()),
        (OpeningCueKind::SettingPhrase, situation_matcher()),
        (OpeningCueKind::SensoryWord, image_matcher()),
        (OpeningCueKind::BackstoryPhrase, backstory_matcher()),
    ] {
        for m in matcher.find_iter(text).take(20_001) {
            found.push((m.start, m.end, kind));
            if found.len() > 20_000 {
                return Err(measurements::invalid("opening exceeds 20000 source cues"));
            }
        }
    }
    static TOPICS: OnceLock<crate::craft::lexicon::WordMatcher> = OnceLock::new();
    for m in TOPICS
        .get_or_init(|| {
            crate::craft::lexicon::WordMatcher::new(&[
                "secret",
                "hidden",
                "danger",
                "threat",
                "warning",
                "mystery",
                "love",
                "attraction",
                "quest",
                "journey",
                "mission",
                "revenge",
                "justice",
                "death",
                "kill",
            ])
        })
        .find_iter(text)
    {
        found.push((m.start, m.end, OpeningCueKind::TopicWord));
        if found.len() > 20_000 {
            return Err(measurements::invalid("opening exceeds 20000 source cues"));
        }
    }
    for (start, c) in text.char_indices() {
        if c == '?' {
            found.push((start, start + 1, OpeningCueKind::QuestionMarker));
            if found.len() > 20_000 {
                return Err(measurements::invalid("opening exceeds 20000 source cues"));
            }
        }
    }
    static CAPITALIZED: OnceLock<regex::Regex> = OnceLock::new();
    for m in CAPITALIZED
        .get_or_init(|| {
            regex::Regex::new(r"\b\p{Lu}[\p{L}\p{M}]*\b")
                .expect("static capitalized-token expression")
        })
        .find_iter(text)
    {
        found.push((m.start(), m.end(), OpeningCueKind::CapitalizedToken));
        if found.len() > 20_000 {
            return Err(measurements::invalid("opening exceeds 20000 source cues"));
        }
    }
    found.sort_by_key(|(start, end, kind)| (*start, *end, *kind));
    let quotes = extract_dialogue_checked(text, 0, &[], &DialogueExtractionOptions::default())?;
    let para_spans: Vec<_> = paragraphs.iter().map(|p| source_span(text, p)).collect();
    let mut spans: Vec<_> = found.iter().map(|(start, end, _)| (*start, *end)).collect();
    spans.extend(&para_spans);
    if let Some(span) = first_line {
        spans.push(span);
    }
    if let Some(span) = first_sentence {
        spans.push(span);
    }
    let mut ranges = measurements::ranges(text, &spans)?.into_iter();
    let cues = found
        .into_iter()
        .map(|(start, end, kind)| OpeningCue {
            kind,
            range: ranges.next().expect("cue span exists"),
            quote: text[start..end].into(),
        })
        .collect();
    let mut quote_starts = quotes
        .lines
        .iter()
        .filter_map(|line| line.quotation_range.as_ref().map(|range| range.byte_start))
        .peekable();
    let paragraphs = paragraphs
        .iter()
        .zip(para_spans)
        .map(|(paragraph, (start, end))| {
            let sentences = crate::substrate::text::split_sentences(paragraph);
            while quote_starts
                .peek()
                .is_some_and(|position| *position < start)
            {
                quote_starts.next();
            }
            let mut quotation_candidate_count = 0;
            while quote_starts.peek().is_some_and(|position| *position < end) {
                quotation_candidate_count += 1;
                quote_starts.next();
            }
            OpeningParagraph {
                range: ranges.next().expect("paragraph span exists"),
                word_count: measurements::tokens(paragraph).len(),
                sentence_count: sentences.len(),
                backstory_sentence_count: sentences
                    .iter()
                    .filter(|sentence| backstory_matcher().is_match(sentence))
                    .count(),
                quotation_candidate_count,
            }
        })
        .collect();
    Ok(MeasuredOpening {
        first_line_range: first_line.and_then(|_| ranges.next()),
        first_sentence_range: first_sentence.and_then(|_| ranges.next()),
        paragraphs,
        cues,
        diagnostics: quotes
            .diagnostics
            .into_iter()
            .map(|d| format!("quotation_candidates:{d}"))
            .collect(),
    })
}
