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

use std::collections::HashMap;

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

/// Round a float to an arbitrary number of decimal places.
#[inline]
pub fn round_to(v: f64, decimals: u32) -> f64 {
    let m = 10_f64.powi(decimals as i32);
    (v * m).round() / m
}

/// Safe division that returns 0.0 when the denominator is zero.
#[inline]
pub fn safe_div(num: f64, den: f64) -> f64 {
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

/// Clamp a value to the unit interval [0.0, 1.0].
#[inline]
pub fn clamp_unit(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

/// Clamp a value to the signed unit interval [-1.0, 1.0].
#[inline]
pub fn clamp_signed(v: f64) -> f64 {
    v.clamp(-1.0, 1.0)
}

/// Compute a percentage: (count / total) * 100.0, returning 0.0 if total is 0.
#[inline]
pub fn percentage(count: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        (count as f64 / total as f64) * 100.0
    }
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

/// Population variance. Returns 0.0 for empty slices.
pub fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let m = mean(values);
    values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / values.len() as f64
}

/// Population standard deviation. Returns 0.0 for empty slices.
pub fn std_dev(values: &[f64]) -> f64 {
    variance(values).sqrt()
}

/// Segment a slice into `n` roughly equal parts and aggregate each with `f`.
/// Returns empty Vec if values is empty or n is 0.
pub fn segment_and_aggregate(values: &[f64], n: usize, f: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    if values.is_empty() || n == 0 {
        return Vec::new();
    }
    let seg_size = (values.len() / n).max(1);
    (0..n)
        .map(|i| {
            let start = (i * seg_size).min(values.len());
            let end = if i == n - 1 {
                values.len()
            } else {
                ((i + 1) * seg_size).min(values.len())
            };
            f(&values[start..end])
        })
        .collect()
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
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Cosine similarity between two `HashMap<String, f64>` frequency maps.
/// Iterates over the union of keys; missing keys default to 0.0.
/// Returns 0.0 when either map is empty or both magnitudes are near-zero.
pub fn map_cosine_similarity(a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut mag_a = 0.0_f64;
    let mut mag_b = 0.0_f64;

    for (word, &freq_a) in a.iter() {
        mag_a += freq_a * freq_a;
        if let Some(&freq_b) = b.get(word) {
            dot += freq_a * freq_b;
        }
    }
    for &freq_b in b.values() {
        mag_b += freq_b * freq_b;
    }

    let denom = mag_a.sqrt() * mag_b.sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        dot / denom
    }
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

/// Euclidean distance between two score maps over a set of keys.
/// Missing keys default to 0.0.
pub fn map_distance(a: &HashMap<String, f64>, b: &HashMap<String, f64>, keys: &[&str]) -> f64 {
    let sum: f64 = keys
        .iter()
        .map(|k| {
            let va = a.get(*k).copied().unwrap_or(0.0);
            let vb = b.get(*k).copied().unwrap_or(0.0);
            (va - vb).powi(2)
        })
        .sum();
    sum.sqrt()
}

/// Pearson correlation between two f64 slices over their common prefix.
/// Returns 0.0 when fewer than 2 paired values or either series has near-zero
/// variance.
pub fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let mean_a: f64 = a[..n].iter().sum::<f64>() / n as f64;
    let mean_b: f64 = b[..n].iter().sum::<f64>() / n as f64;
    let mut cov = 0.0_f64;
    let mut var_a = 0.0_f64;
    let mut var_b = 0.0_f64;
    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    cov / denom
}

/// Kullback-Leibler divergence D(p || q) in nats. Each element is floored to a
/// small epsilon to avoid log(0)/division by zero.
///
/// # Contract
/// `p` and `q` MUST have equal length: they are two distributions over the same
/// support. A length mismatch is always a caller bug.
///
/// - In debug/test builds this misuse panics via `debug_assert_eq!`, so it is
///   caught loudly at its source rather than silently producing a wrong number.
/// - In release builds (where the assert is compiled out) the defined fallback
///   is to iterate the common prefix (`p.len().min(q.len())` via `zip`) and
///   ignore the trailing elements of the longer slice. This keeps the function
///   total (it never panics in release) and deterministic; the result is the KL
///   over the overlapping support. This is a damage-limiting fallback for a
///   condition that should never occur, NOT a supported mode of use.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    debug_assert_eq!(
        p.len(),
        q.len(),
        "kl_divergence: p and q must have equal length (distributions over the same support); \
         got p.len()={} q.len()={}",
        p.len(),
        q.len()
    );
    const EPS: f64 = 1e-10;
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            let pi = pi.max(EPS);
            let qi = qi.max(EPS);
            pi * (pi / qi).ln()
        })
        .sum()
}

// ===========================================================================
// Text processing
// ===========================================================================

/// Truncate a string to at most `max_len` bytes on a UTF-8 char boundary.
/// Returns the input unchanged when it already fits. Appends no ellipsis.
pub fn truncate_str(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        return s;
    }
    let end = crate::substrate::text::floor_char_boundary(s, max_len);
    &s[..end]
}

/// Sort `(item, score)` pairs by score descending, NaN-safe. The one home for
/// the rank-by-similarity idiom used across embeddings, artifacts, and graph.
pub fn sort_by_score_desc<T>(items: &mut [(T, f64)]) {
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
}

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

/// Split text into sentences using punctuation boundaries.
/// Handles abbreviations, ellipses, and multi-punctuation clusters.
pub fn split_sentences(text: &str) -> Vec<String> {
    sentence_spans(text)
        .into_iter()
        .map(|(s, e)| text[s..e].to_string())
        .collect()
}

/// Abbreviation-aware sentence segmentation returning trimmed `(start, end)`
/// byte ranges into `text`. The single robust sentence boundary primitive:
/// owned/slice splitters, counting, and start-position lookups all build on it,
/// so nothing re-rolls the fragile `[.!?]\s+` regex (which breaks on "Dr.",
/// "3.14", "U.S.A.", ellipses).
pub fn sentence_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut start = 0;
    let mut i = 0;

    while i < len {
        if matches!(bytes[i], b'.' | b'!' | b'?') {
            // End of the punctuation cluster (e.g. "..." or "?!").
            let mut j = i + 1;
            while j < len && matches!(bytes[j], b'.' | b'!' | b'?') {
                j += 1;
            }
            // Single period after a common abbreviation: not a boundary.
            if bytes[i] == b'.' && j == i + 1 && is_abbreviation_before(text, i) {
                i = j;
                continue;
            }
            // Boundary only when followed by whitespace (or end of text).
            if j >= len || bytes[j].is_ascii_whitespace() {
                if let Some(span) = trimmed_span(text, start, j) {
                    spans.push(span);
                }
                while j < len && bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
                start = j;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    if let Some(span) = trimmed_span(text, start, len) {
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
    if s < e {
        Some((s, e))
    } else {
        None
    }
}

/// Check if the character before position `dot_pos` ends a common abbreviation.
fn is_abbreviation_before(text: &str, dot_pos: usize) -> bool {
    static ABBREVIATIONS: &[&str] = &[
        "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St", "Ave", "Blvd", "vs", "etc", "approx",
        "dept", "est", "vol", "Gen", "Gov", "Sgt", "Cpl", "Pvt", "Lt", "Col", "Capt", "Maj", "Rev",
        "Hon",
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
    ABBREVIATIONS.contains(&word)
}

// ===========================================================================
// Iterator utilities
// ===========================================================================

/// Find the index of the maximum value in a float slice.
/// Returns 0 for empty slices. Uses total ordering (NaN-safe).
pub fn max_index(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Find the index of the minimum value in a float slice.
/// Returns 0 for empty slices. Uses total ordering (NaN-safe).
pub fn min_index(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Increment a counter in a HashMap, inserting 0 as default.
#[inline]
pub fn increment(map: &mut HashMap<String, usize>, key: &str) {
    *map.entry(key.to_string()).or_insert(0) += 1;
}

/// Increment a float counter in a HashMap, inserting 0.0 as default.
#[inline]
pub fn increment_f64(map: &mut HashMap<String, f64>, key: &str, amount: f64) {
    *map.entry(key.to_string()).or_insert(0.0) += amount;
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
// Genre classification
// ===========================================================================

/// Canonical genre enum that consolidates the many string aliases used
/// across analysis modules.  Use `Genre::from("thriller")` or
/// `Genre::from(some_string.as_str())` to parse user-supplied strings.
///
/// All comparisons are case-insensitive.  Unknown strings map to `General`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Genre {
    Thriller,
    Horror,
    Romance,
    Literary,
    Mystery,
    SciFi,
    Fantasy,
    Comedy,
    Historical,
    Drama,
    YoungAdult,
    MiddleGrade,
    Memoir,
    WomensFiction,
    General,
}

impl Genre {
    /// All concrete (non-General) variants in definition order.
    pub const ALL: &'static [Genre] = &[
        Genre::Thriller,
        Genre::Horror,
        Genre::Romance,
        Genre::Literary,
        Genre::Mystery,
        Genre::SciFi,
        Genre::Fantasy,
        Genre::Comedy,
        Genre::Historical,
        Genre::Drama,
        Genre::YoungAdult,
        Genre::MiddleGrade,
        Genre::Memoir,
        Genre::WomensFiction,
    ];

    /// Returns `true` if the given free-form genre string resolves to this
    /// variant (case-insensitive, alias-aware).
    pub fn matches(&self, genre_str: &str) -> bool {
        *self == Genre::from(genre_str)
    }

    /// Canonical lowercase label suitable for display or serialisation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Genre::Thriller => "thriller",
            Genre::Horror => "horror",
            Genre::Romance => "romance",
            Genre::Literary => "literary",
            Genre::Mystery => "mystery",
            Genre::SciFi => "sci-fi",
            Genre::Fantasy => "fantasy",
            Genre::Comedy => "comedy",
            Genre::Historical => "historical",
            Genre::Drama => "drama",
            Genre::YoungAdult => "young adult",
            Genre::MiddleGrade => "middle grade",
            Genre::Memoir => "memoir",
            Genre::WomensFiction => "women's fiction",
            Genre::General => "general",
        }
    }
}

impl From<&str> for Genre {
    fn from(s: &str) -> Self {
        let lowered = s.to_lowercase();
        match lowered.trim() {
            "thriller" | "suspense" => Genre::Thriller,
            "horror" => Genre::Horror,
            "romance" => Genre::Romance,
            "literary" | "literary fiction" => Genre::Literary,
            "mystery" | "detective" => Genre::Mystery,
            "sci-fi" | "science fiction" | "scifi" => Genre::SciFi,
            "fantasy" | "epic fantasy" | "urban fantasy" => Genre::Fantasy,
            "comedy" | "humor" | "humour" => Genre::Comedy,
            "historical" | "historical fiction" => Genre::Historical,
            "drama" => Genre::Drama,
            "ya" | "young adult" => Genre::YoungAdult,
            "middle grade" | "mg" => Genre::MiddleGrade,
            "memoir" => Genre::Memoir,
            "women's fiction" => Genre::WomensFiction,
            _ => Genre::General,
        }
    }
}

impl std::fmt::Display for Genre {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
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

/// Linear interpolation; `t` is clamped to [0.0, 1.0].
#[inline]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Rescale a slice into [0.0, 1.0]. Returns all-zeros for empty or constant input.
pub fn min_max_scale(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let lo = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    values.iter().map(|&v| normalize(v, lo, hi)).collect()
}

// ===========================================================================
// Statistics (extended)
// ===========================================================================

/// Median. Returns 0.0 for empty slices.
///
// PERF: clones and fully sorts `values` on every call. Signature takes `&[f64]`
// (callers retain ownership and may reuse the unsorted slice), so we cannot sort
// in place. Computing many quantiles over one dataset re-sorts each time; if
// that ever shows up in a profile, add a `*_sorted(&[f64])` variant that assumes
// pre-sorted input rather than changing this signature.
pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut s = values.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = s.len() / 2;
    if s.len().is_multiple_of(2) {
        (s[mid - 1] + s[mid]) / 2.0
    } else {
        s[mid]
    }
}

/// Linear-interpolated percentile (`p` in [0, 100]). Returns 0.0 for empty slices.
///
// PERF: clones and fully sorts `values` on every call (same trade-off as
// `median`). Requesting several percentiles of one dataset re-sorts per call;
// if a profile flags this, add a `percentile_sorted(&[f64], f64)` variant that
// assumes pre-sorted input rather than changing this signature.
pub fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut s = values.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = p.clamp(0.0, 100.0) / 100.0 * (s.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        s[lo]
    } else {
        let frac = rank - lo as f64;
        s[lo] * (1.0 - frac) + s[hi] * frac
    }
}

/// Weighted mean. Returns 0.0 if empty, length-mismatched, or total weight is 0.
pub fn weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    if values.is_empty() || values.len() != weights.len() {
        return 0.0;
    }
    let total: f64 = weights.iter().sum();
    if total == 0.0 {
        return 0.0;
    }
    values.iter().zip(weights).map(|(v, w)| v * w).sum::<f64>() / total
}

/// Z-scores. Returns all-zeros for empty or zero-variance input.
pub fn z_scores(values: &[f64]) -> Vec<f64> {
    let sd = std_dev(values);
    if values.is_empty() || sd == 0.0 {
        return vec![0.0; values.len()];
    }
    let m = mean(values);
    values.iter().map(|&v| (v - m) / sd).collect()
}

/// Trailing moving average; a window of 0 or 1 returns a copy. Empty for empty
/// input. O(n) via a rolling sum.
pub fn moving_average(values: &[f64], window: usize) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let w = window.max(1);
    let mut out = Vec::with_capacity(values.len());
    let mut sum = 0.0;
    for i in 0..values.len() {
        sum += values[i];
        if i >= w {
            sum -= values[i - w];
        }
        out.push(sum / (i + 1).min(w) as f64);
    }
    out
}

// ===========================================================================
// Selection
// ===========================================================================

/// Indices of the top `k` values, descending; ties broken by index.
pub fn top_k_indices(values: &[f64], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_sentences_multibyte_whitespace_before_period() {
        // Regression: NEL (`\u{85}`, a 2-byte Windows-1252 control) immediately
        // before an abbreviation made the word-boundary scan slice inside the
        // character and panic, blocking real manuscripts from loading.
        let text = "See the list.\u{85}Dr. Smith arrived. The end.";
        let sents = split_sentences(text);
        assert!(!sents.is_empty());
        assert!(sents.iter().any(|s| s.contains("Dr. Smith arrived")));
    }

    #[test]
    fn test_slug() {
        assert_eq!(slug("My Novel"), "my-novel");
        assert_eq!(slug("  A B  C "), "a-b-c");
        assert_eq!(slug("Café/Bar #1"), "caf-bar-1");
        assert_eq!(slug("***"), "untitled");
    }

    #[test]
    fn test_normalize_and_lerp() {
        assert_eq!(normalize(5.0, 0.0, 10.0), 0.5);
        assert_eq!(normalize(20.0, 0.0, 10.0), 1.0);
        assert_eq!(normalize(5.0, 5.0, 5.0), 0.0);
        assert_eq!(lerp(0.0, 10.0, 0.25), 2.5);
        assert_eq!(lerp(0.0, 10.0, 2.0), 10.0);
    }

    #[test]
    fn test_min_max_scale() {
        assert_eq!(min_max_scale(&[2.0, 4.0, 6.0]), vec![0.0, 0.5, 1.0]);
        assert_eq!(min_max_scale(&[3.0, 3.0]), vec![0.0, 0.0]);
        assert!(min_max_scale(&[]).is_empty());
    }

    #[test]
    fn test_median_and_percentile() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median(&[]), 0.0);
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 50.0), 2.5);
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 0.0), 1.0);
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 100.0), 4.0);
    }

    #[test]
    fn test_weighted_mean() {
        assert_eq!(weighted_mean(&[1.0, 3.0], &[1.0, 1.0]), 2.0);
        assert_eq!(weighted_mean(&[1.0, 3.0], &[3.0, 1.0]), 1.5);
        assert_eq!(weighted_mean(&[1.0], &[0.0]), 0.0);
    }

    #[test]
    fn test_z_scores() {
        let z = z_scores(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((z[0] - (-1.5)).abs() < 0.01);
        assert_eq!(z_scores(&[3.0, 3.0]), vec![0.0, 0.0]);
    }

    #[test]
    fn test_moving_average() {
        assert_eq!(moving_average(&[1.0, 2.0, 3.0], 1), vec![1.0, 2.0, 3.0]);
        assert_eq!(moving_average(&[2.0, 4.0, 6.0], 2), vec![2.0, 3.0, 5.0]);
    }

    #[test]
    fn test_top_k_indices() {
        assert_eq!(top_k_indices(&[0.1, 0.9, 0.5, 0.3], 2), vec![1, 2]);
        assert_eq!(top_k_indices(&[1.0, 1.0, 1.0], 2), vec![0, 1]);
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
    fn test_round_to() {
        assert_eq!(round_to(1.23456, 3), 1.235);
        assert_eq!(round_to(1.23456, 0), 1.0);
    }

    #[test]
    fn test_safe_div() {
        assert_eq!(safe_div(10.0, 2.0), 5.0);
        assert_eq!(safe_div(10.0, 0.0), 0.0);
    }

    #[test]
    fn test_clamp_unit() {
        assert_eq!(clamp_unit(1.5), 1.0);
        assert_eq!(clamp_unit(-0.5), 0.0);
        assert_eq!(clamp_unit(0.5), 0.5);
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
    }

    #[test]
    fn test_variance() {
        assert_eq!(variance(&[]), 0.0);
        assert_eq!(variance(&[5.0, 5.0, 5.0]), 0.0);
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
    fn test_map_cosine_similarity() {
        let a: HashMap<String, f64> = HashMap::new();
        let b: HashMap<String, f64> = HashMap::new();
        assert_eq!(map_cosine_similarity(&a, &b), 0.0);

        let a: HashMap<String, f64> = [("x".into(), 1.0), ("y".into(), 0.0)].into();
        let b: HashMap<String, f64> = [("x".into(), 1.0), ("y".into(), 0.0)].into();
        assert!((map_cosine_similarity(&a, &b) - 1.0).abs() < 1e-10);

        let a: HashMap<String, f64> = [("x".into(), 1.0)].into();
        let b: HashMap<String, f64> = [("y".into(), 1.0)].into();
        assert_eq!(map_cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_euclidean_distance() {
        let d = euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_split_sentences() {
        let sents = split_sentences("Hello world. How are you? I'm fine!");
        assert_eq!(sents.len(), 3);
        assert_eq!(sents[0], "Hello world.");
        assert_eq!(sents[1], "How are you?");
    }

    #[test]
    fn test_split_sentences_abbreviation() {
        let sents = split_sentences("Mr. Smith went home. He was tired.");
        assert_eq!(sents.len(), 2);
    }

    #[test]
    fn test_max_index() {
        assert_eq!(max_index(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(max_index(&[]), 0);
    }

    #[test]
    fn test_min_index() {
        assert_eq!(min_index(&[3.0, 1.0, 2.0]), 1);
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
    fn test_kl_divergence_values() {
        // Identical distributions -> 0.
        let p = [0.25, 0.25, 0.25, 0.25];
        assert!(kl_divergence(&p, &p).abs() < 1e-12);
        // Known value: D([0.5,0.5] || [0.25,0.75]) in nats.
        let kl = kl_divergence(&[0.5, 0.5], &[0.25, 0.75]);
        let expected = 0.5 * (0.5_f64 / 0.25).ln() + 0.5 * (0.5_f64 / 0.75).ln();
        assert!((kl - expected).abs() < 1e-12);
        // Asymmetric: D(p||q) != D(q||p) in general.
        let a = [0.7, 0.3];
        let b = [0.4, 0.6];
        assert!((kl_divergence(&a, &b) - kl_divergence(&b, &a)).abs() > 1e-6);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "equal length")]
    fn test_kl_divergence_length_mismatch_panics_in_debug() {
        let _ = kl_divergence(&[0.5, 0.5], &[1.0]);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "highest to lowest")]
    fn test_classify_descending_unsorted_panics_in_debug() {
        let _ = classify_descending(0.5, &[(0.4, "medium"), (0.7, "high")], "low");
    }

    #[test]
    fn test_segment_and_aggregate() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = segment_and_aggregate(&vals, 3, mean);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.5).abs() < 1e-10);
        assert!((result[1] - 3.5).abs() < 1e-10);
        assert!((result[2] - 5.5).abs() < 1e-10);
    }

    #[test]
    fn segment_and_aggregate_handles_more_segments_than_values() {
        let r = segment_and_aggregate(&[1.0, 2.0, 3.0], 5, mean);
        assert_eq!(r.len(), 5);
        assert_eq!(r[0], 1.0);
        assert_eq!(r[4], 0.0);
    }

    #[test]
    fn test_percentage() {
        assert_eq!(percentage(1, 4), 25.0);
        assert_eq!(percentage(0, 0), 0.0);
    }

    #[test]
    fn test_increment() {
        let mut map = HashMap::new();
        increment(&mut map, "a");
        increment(&mut map, "a");
        increment(&mut map, "b");
        assert_eq!(map["a"], 2);
        assert_eq!(map["b"], 1);
    }

    #[test]
    fn test_genre_from_str_canonical() {
        assert_eq!(Genre::from("thriller"), Genre::Thriller);
        assert_eq!(Genre::from("horror"), Genre::Horror);
        assert_eq!(Genre::from("romance"), Genre::Romance);
        assert_eq!(Genre::from("literary"), Genre::Literary);
        assert_eq!(Genre::from("mystery"), Genre::Mystery);
        assert_eq!(Genre::from("sci-fi"), Genre::SciFi);
        assert_eq!(Genre::from("fantasy"), Genre::Fantasy);
        assert_eq!(Genre::from("comedy"), Genre::Comedy);
        assert_eq!(Genre::from("historical"), Genre::Historical);
        assert_eq!(Genre::from("drama"), Genre::Drama);
        assert_eq!(Genre::from("ya"), Genre::YoungAdult);
        assert_eq!(Genre::from("middle grade"), Genre::MiddleGrade);
        assert_eq!(Genre::from("memoir"), Genre::Memoir);
        assert_eq!(Genre::from("women's fiction"), Genre::WomensFiction);
    }

    #[test]
    fn test_genre_from_str_aliases() {
        assert_eq!(Genre::from("suspense"), Genre::Thriller);
        assert_eq!(Genre::from("literary fiction"), Genre::Literary);
        assert_eq!(Genre::from("detective"), Genre::Mystery);
        assert_eq!(Genre::from("science fiction"), Genre::SciFi);
        assert_eq!(Genre::from("scifi"), Genre::SciFi);
        assert_eq!(Genre::from("epic fantasy"), Genre::Fantasy);
        assert_eq!(Genre::from("urban fantasy"), Genre::Fantasy);
        assert_eq!(Genre::from("humor"), Genre::Comedy);
        assert_eq!(Genre::from("humour"), Genre::Comedy);
        assert_eq!(Genre::from("historical fiction"), Genre::Historical);
        assert_eq!(Genre::from("young adult"), Genre::YoungAdult);
        assert_eq!(Genre::from("mg"), Genre::MiddleGrade);
    }

    #[test]
    fn test_genre_case_insensitive() {
        assert_eq!(Genre::from("THRILLER"), Genre::Thriller);
        assert_eq!(Genre::from("Romance"), Genre::Romance);
        assert_eq!(Genre::from("Sci-Fi"), Genre::SciFi);
        assert_eq!(Genre::from("LITERARY FICTION"), Genre::Literary);
        assert_eq!(Genre::from("Young Adult"), Genre::YoungAdult);
    }

    #[test]
    fn test_genre_unknown_maps_to_general() {
        assert_eq!(Genre::from("western"), Genre::General);
        assert_eq!(Genre::from(""), Genre::General);
        assert_eq!(Genre::from("nonfiction"), Genre::General);
    }

    #[test]
    fn test_genre_matches() {
        assert!(Genre::Thriller.matches("thriller"));
        assert!(Genre::Thriller.matches("Suspense"));
        assert!(!Genre::Thriller.matches("romance"));
        assert!(Genre::SciFi.matches("science fiction"));
        assert!(Genre::General.matches("unknown genre"));
    }

    #[test]
    fn test_genre_as_str_roundtrip() {
        for &g in Genre::ALL {
            assert_eq!(Genre::from(g.as_str()), g);
        }
    }

    #[test]
    fn test_genre_display() {
        assert_eq!(format!("{}", Genre::Thriller), "thriller");
        assert_eq!(format!("{}", Genre::SciFi), "sci-fi");
        assert_eq!(format!("{}", Genre::WomensFiction), "women's fiction");
    }
}
