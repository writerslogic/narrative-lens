//! Shared utility functions for narrative-lens analysis modules.
//!
//! This is the **canonical home** for math, statistics, similarity, and small
//! iterator/collection helpers. Before writing an inline `min`/`max`/clamp,
//! mean, normalization, or similarity computation in an analyzer, use one of
//! these. If the helper you need is missing, add it here rather than locally, so
//! it is shared and tested once.
//!
//! Canonical homes (do not duplicate across modules):
//! - numeric / statistics / similarity / selection -> this module (`utils`)
//! - word/sentence tokenizing, syllables, counting, normalization -> `text`
//! - higher-level linguistic ops (tf-idf, stylometry, NER) -> `nlp`
//! - small shared data types -> `common_types`

// ===========================================================================
// Math helpers
// ===========================================================================

/// Round a float to 1 decimal place.
#[inline]
pub fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

/// Round a float to 2 decimal places.
#[inline]
pub fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// Round a float to 3 decimal places.
#[inline]
pub fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

/// Round a float to 4 decimal places.
#[inline]
pub fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

/// Safe division that returns 0.0 when the denominator is zero.
#[inline]
pub fn safe_div(num: f64, den: f64) -> f64 {
    if den == 0.0 { 0.0 } else { num / den }
}

// ===========================================================================
// Statistics
// ===========================================================================

/// Arithmetic mean. Returns 0.0 for empty slices.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Population standard deviation. Returns 0.0 for empty slices.
pub fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let m = mean(values);
    (values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
}

// ===========================================================================
// Similarity / distance
// ===========================================================================

/// Cosine similarity between two f64 slices. Returns 0.0 for empty or mismatched inputs.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// Euclidean distance between two f64 slices. Returns 0.0 for empty or mismatched inputs.
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ===========================================================================
// Text processing
// ===========================================================================

/// Path- and id-safe slug from arbitrary text: ASCII alphanumerics lowercased,
/// every other run collapsed to a single `-`, trimmed. Falls back to a stable
/// non-empty default so the result is always a usable identifier.
pub fn slug(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_dash = false;
    for c in text.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    let trimmed = out.trim_matches('-');
    if trimmed.is_empty() {
        "untitled".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Abbreviation-aware sentence segmentation returning trimmed `(start, end)`
/// byte ranges into `text`. The single robust sentence boundary primitive:
/// owned/slice splitters, counting, and start-position lookups all build on it,
/// so nothing re-rolls the fragile `[.!?]\s+` regex (which breaks on "Dr.",
/// "3.14", "U.S.A.", ellipses).
pub fn sentence_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((position, ch)) = chars.next() {
        if !matches!(ch, '.' | '!' | '?' | '\u{2026}') {
            continue;
        }
        let mut end = position + ch.len_utf8();
        while let Some(&(next, punctuation)) = chars.peek() {
            if !matches!(punctuation, '.' | '!' | '?' | '\u{2026}') {
                break;
            }
            end = next + punctuation.len_utf8();
            chars.next();
        }
        if ch == '.' && end == position + 1 && is_abbreviation_before(text, position) {
            continue;
        }
        while let Some(&(next, closing)) = chars.peek() {
            if !matches!(closing, '\"' | '\'' | '\u{2019}' | '\u{201d}' | ')' | ']') {
                break;
            }
            end = next + closing.len_utf8();
            chars.next();
        }
        if chars.peek().is_none_or(|(_, next)| next.is_whitespace()) {
            if let Some(span) = trimmed_span(text, start, end) {
                spans.push(span);
            }
            while let Some(&(next, whitespace)) = chars.peek() {
                if !whitespace.is_whitespace() {
                    break;
                }
                end = next + whitespace.len_utf8();
                chars.next();
            }
            start = end;
        }
    }
    if let Some(span) = trimmed_span(text, start, text.len()) {
        spans.push(span);
    }
    spans
}

/// Trim leading/trailing whitespace from `text[start..end]` and return the
/// resulting byte range, or `None` if it is empty after trimming.
fn trimmed_span(text: &str, start: usize, end: usize) -> Option<(usize, usize)> {
    let seg = &text[start..end];
    let lead = seg.len() - seg.trim_start().len();
    let trail = seg.len() - seg.trim_end().len();
    let s = start + lead;
    let e = end - trail;
    if s < e { Some((s, e)) } else { None }
}

/// Whether `word` is an honorific or title that precedes a personal name
/// without being part of any name (`Mrs.`, `Sir`, `Colonel`). Case-insensitive;
/// a trailing period is ignored.
pub fn is_honorific(word: &str) -> bool {
    const HONORIFICS: &[&str] = &[
        "mr",
        "mrs",
        "miss",
        "ms",
        "mme",
        "mlle",
        "dr",
        "sir",
        "lady",
        "lord",
        "madam",
        "madame",
        "monsieur",
        "colonel",
        "captain",
        "major",
        "general",
        "sergeant",
        "professor",
        "aunt",
        "uncle",
        "king",
        "queen",
        "prince",
        "princess",
        "judge",
        "reverend",
        "rev",
        "st",
        "saint",
    ];
    HONORIFICS.contains(&word.trim_end_matches('.').to_lowercase().as_str())
}

/// Check if the character before position `dot_pos` ends a common abbreviation.
fn is_abbreviation_before(text: &str, dot_pos: usize) -> bool {
    static ABBREVIATIONS: &[&str] = &[
        "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St", "Ave", "Blvd", "vs", "etc", "approx",
        "dept", "est", "vol", "Gen", "Gov", "Sgt", "Cpl", "Pvt", "Lt", "Col", "Capt", "Maj", "Rev",
        "Hon", "Inc", "Ltd", "Corp", "Vol", "e.g", "i.e", "a.m", "p.m", "U.S.A", "U.S", "U.K",
    ];
    // Find the word immediately before the dot. `rfind` returns the *start*
    // byte of the delimiter char, so advance to the next char boundary rather
    // than a raw +1 — multibyte whitespace (e.g. NEL `\u{85}` from Windows-1252
    // text, non-breaking spaces) would otherwise leave `word_start` inside a
    // character and panic on the slice below.
    let before = &text[..dot_pos];
    let word_start = before
        .rfind(|c: char| c.is_whitespace() || c == '"' || c == '\'')
        .map(|p| crate::substrate::text::ceil_char_boundary(before, p + 1))
        .unwrap_or(0);
    let word = &before[word_start..];
    let sentence_final_abbreviation = ["a.m", "p.m", "U.S", "U.K", "U.S.A"]
        .iter()
        .any(|abbreviation| word.eq_ignore_ascii_case(abbreviation));
    if sentence_final_abbreviation
        && text[dot_pos + 1..]
            .trim_start()
            .chars()
            .next()
            .is_some_and(char::is_uppercase)
    {
        return false;
    }
    ABBREVIATIONS
        .iter()
        .any(|abbreviation| word.eq_ignore_ascii_case(abbreviation))
}

// ===========================================================================
// Classification
// ===========================================================================

/// Classify a score against descending thresholds.
///
/// # Contract
/// `thresholds` MUST be sorted by threshold value from highest to lowest
/// (non-strictly descending). The function scans front-to-back and returns the
/// label of the FIRST entry whose threshold the score meets (`score >=
/// threshold`), so an unsorted list silently returns the wrong bucket. That
/// misuse panics via `debug_assert!` in debug/test builds; in release builds the
/// scan still runs (the assert is compiled out) and yields whatever the
/// first-match-wins traversal produces over the given order.
///
/// Returns the label of the first threshold the score meets, or `fallback` when
/// it meets none.
///
/// Example:
/// ```
/// use narrative_lens::substrate::utils::classify_descending;
/// let level = classify_descending(0.6, &[(0.7, "high"), (0.4, "medium")], "low");
/// assert_eq!(level, "medium");
/// ```
pub fn classify_descending<'a>(
    score: f64,
    thresholds: &[(f64, &'a str)],
    fallback: &'a str,
) -> &'a str {
    debug_assert!(
        thresholds.windows(2).all(|w| w[0].0 >= w[1].0),
        "classify_descending: thresholds must be sorted from highest to lowest; got {:?}",
        thresholds.iter().map(|&(t, _)| t).collect::<Vec<_>>()
    );
    for &(threshold, label) in thresholds {
        if score >= threshold {
            return label;
        }
    }
    fallback
}

// ===========================================================================
// Normalization & interpolation
// ===========================================================================

/// Rescale `v` from [lo, hi] into [0.0, 1.0], clamped. Returns 0.0 if hi <= lo.
#[inline]
pub fn normalize(v: f64, lo: f64, hi: f64) -> f64 {
    if hi <= lo {
        return 0.0;
    }
    ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_spans_multibyte_whitespace_before_period() {
        // Regression: NEL (`\u{85}`, a 2-byte Windows-1252 control) immediately
        // before an abbreviation made the word-boundary scan slice inside the
        // character and panic, blocking real manuscripts from loading.
        let text = "See the list.\u{85}Dr. Smith arrived. The end.";
        let spans = sentence_spans(text);
        assert!(!spans.is_empty());
        assert!(
            spans
                .iter()
                .any(|&(s, e)| text[s..e].contains("Dr. Smith arrived"))
        );
    }

    #[test]
    fn test_sentence_spans_basic_and_abbreviation() {
        let text = "Hello world. How are you? I'm fine!";
        let spans = sentence_spans(text);
        assert_eq!(spans.len(), 3);
        assert_eq!(&text[spans[0].0..spans[0].1], "Hello world.");
        assert_eq!(&text[spans[1].0..spans[1].1], "How are you?");
        assert_eq!(
            sentence_spans("Mr. Smith went home. He was tired.").len(),
            2
        );
    }

    #[test]
    fn test_slug() {
        assert_eq!(slug("My Novel"), "my-novel");
        assert_eq!(slug("  A B  C "), "a-b-c");
        assert_eq!(slug("Café/Bar #1"), "caf-bar-1");
        assert_eq!(slug("***"), "untitled");
    }

    #[test]
    fn test_normalize() {
        assert_eq!(normalize(5.0, 0.0, 10.0), 0.5);
        assert_eq!(normalize(20.0, 0.0, 10.0), 1.0);
        assert_eq!(normalize(5.0, 5.0, 5.0), 0.0);
    }

    #[test]
    fn test_round2() {
        assert_eq!(round2(0.12345), 0.12);
        assert_eq!(round2(0.999), 1.0);
        assert_eq!(round2(0.005), 0.01);
    }

    #[test]
    fn test_round4() {
        assert_eq!(round4(0.123456), 0.1235);
    }

    #[test]
    fn test_safe_div() {
        assert_eq!(safe_div(10.0, 2.0), 5.0);
        assert_eq!(safe_div(10.0, 0.0), 0.0);
    }

    #[test]
    fn test_mean() {
        assert_eq!(mean(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn test_std_dev() {
        let sd = std_dev(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((sd - 2.0).abs() < 0.01);
        assert_eq!(std_dev(&[]), 0.0);
        assert_eq!(std_dev(&[5.0, 5.0, 5.0]), 0.0);
    }

    #[test]
    fn test_cosine_similarity() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        let sim = cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((sim - 1.0).abs() < 1e-10);
        let sim2 = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(sim2.abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance() {
        let d = euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_classify_descending() {
        assert_eq!(
            classify_descending(0.8, &[(0.7, "high"), (0.4, "medium")], "low"),
            "high"
        );
        assert_eq!(
            classify_descending(0.5, &[(0.7, "high"), (0.4, "medium")], "low"),
            "medium"
        );
        assert_eq!(
            classify_descending(0.2, &[(0.7, "high"), (0.4, "medium")], "low"),
            "low"
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "highest to lowest")]
    fn test_classify_descending_unsorted_panics_in_debug() {
        let _ = classify_descending(0.5, &[(0.4, "medium"), (0.7, "high")], "low");
    }
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisAvailability {
    Complete,
    Partial,
    #[default]
    Unavailable,
}
