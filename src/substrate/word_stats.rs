//! Word frequency / repetition stats.
//!
//! The most-frequent *content* words across the manuscript (or one document),
//! ranked, for repetition detection and pacing — a writer's tool ("you've leaned
//! on 'suddenly' 41 times"), not a grade of the prose: this surfaces the
//! writer's own patterns, it never judges them.
//!
//! This is deliberately *not* [`crate::substrate::wordfreq`]. That module is a
//! static Zipf lexicon (a word's rank in general English, for
//! rarity/difficulty scoring); this one counts the writer's actual usage in
//! *their* text. Same idea of "frequency", opposite corpus — hence the
//! separate name so the two never get confused at a call site.
//!
//! The counting core is a pure projection: it is computed on demand from live
//! text and never persisted, so it cannot drift from the manuscript. Output is
//! fully deterministic — ties (equal counts) break alphabetically — so the same
//! text always yields the same ranking, which keeps snapshots and the UI stable.

use serde::{Deserialize, Serialize};

/// One word and how many times it occurs in the analyzed text. `Eq` is sound
/// here (only a `String` and a `usize`; no floats), unlike float-carrying
/// payloads elsewhere.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WordCount {
    /// The content word, lowercased.
    pub word: String,
    /// How many times it occurs in the analyzed text.
    pub count: usize,
}

/// Rank the most-frequent content words in `text`, most-frequent first, keeping
/// at most `limit`.
///
/// Pure and deterministic. Tokens are lowercased and split on word boundaries;
/// stopwords (per the caller's `is_stopword` predicate) and single-character
/// tokens are dropped, since neither carries repetition signal a writer would
/// act on. The survivors are counted and sorted by **count descending, then word
/// ascending** — the alphabetical tie-break is what makes the ordering total and
/// reproducible (a `HashMap`'s iteration order is not). `limit` is applied
/// literally: `0` yields an empty result; callers are expected to clamp
/// `limit` to a sane range before calling.
///
/// **Tokenization:** this standalone entry point splits on any non-alphanumeric
/// character, which is dependency-free and good enough for unit tests. A
/// caller that already has tokens can instead tokenize with
/// [`crate::substrate::text::tokenize_words`] (which preserves internal
/// apostrophes and matches the crate's other word counts) and call
/// [`word_frequencies_from_words`] directly, so both paths share the one
/// counting/ranking core below and can't diverge.
pub fn word_frequencies(
    text: &str,
    limit: usize,
    is_stopword: impl Fn(&str) -> bool,
) -> Vec<WordCount> {
    let words = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase);
    word_frequencies_from_words(words, limit, is_stopword)
}

/// Count and rank already-tokenized `words`, applying the same content-word
/// filter, ordering, and `limit` as [`word_frequencies`]. This is the shared
/// core: the `&str` wrapper above feeds it boundary-split tokens, while a
/// caller with pre-tokenized text can feed it
/// [`crate::substrate::text::tokenize_words`] output directly, so the ranking
/// rule lives in exactly one place.
///
/// Words are lowercased defensively (so a caller that passes mixed-case tokens
/// still gets correct counts), then single-character and stopword tokens are
/// dropped before counting.
pub fn word_frequencies_from_words(
    words: impl IntoIterator<Item = String>,
    limit: usize,
    is_stopword: impl Fn(&str) -> bool,
) -> Vec<WordCount> {
    use std::collections::HashMap;

    let mut counts: HashMap<String, usize> = HashMap::new();
    for word in words {
        let word = word.to_lowercase();
        // Single-character tokens (and stopwords) carry no repetition signal a
        // writer would act on, so they never enter the tally. Count chars, not
        // bytes, so a one-glyph multibyte token is still treated as single-char.
        if word.chars().count() <= 1 || is_stopword(&word) {
            continue;
        }
        *counts.entry(word).or_insert(0) += 1;
    }

    let mut ranked: Vec<WordCount> = counts
        .into_iter()
        .map(|(word, count)| WordCount { word, count })
        .collect();
    // Count descending, then word ascending: the alphabetical tie-break makes the
    // order total and reproducible across runs (HashMap order is not stable).
    ranked.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.word.cmp(&b.word)));
    ranked.truncate(limit);
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// A closure-friendly stopword predicate over a fixed set, so the tests carry
    /// no dependency on `crate::substrate::text::stopwords`.
    fn stopwords(words: &[&str]) -> impl Fn(&str) -> bool {
        let set: HashSet<String> = words.iter().map(|s| s.to_string()).collect();
        move |w: &str| set.contains(w)
    }

    #[test]
    fn repeated_words_are_counted_and_ranked() {
        let text = "storm storm storm rain rain wind";
        let got = word_frequencies(text, 10, |_| false);
        assert_eq!(
            got,
            vec![
                WordCount {
                    word: "storm".into(),
                    count: 3
                },
                WordCount {
                    word: "rain".into(),
                    count: 2
                },
                WordCount {
                    word: "wind".into(),
                    count: 1
                },
            ]
        );
    }

    #[test]
    fn stopwords_are_excluded() {
        // "the" and "of" are stopwords; only content words survive.
        let text = "the harbor of the harbor of mist";
        let got = word_frequencies(text, 10, stopwords(&["the", "of"]));
        assert_eq!(
            got,
            vec![
                WordCount {
                    word: "harbor".into(),
                    count: 2
                },
                WordCount {
                    word: "mist".into(),
                    count: 1
                },
            ]
        );
    }

    #[test]
    fn ties_break_alphabetically() {
        // All three occur once; the tie-break orders them a < b < c, regardless
        // of input order.
        let text = "cedar birch ash";
        let got = word_frequencies(text, 10, |_| false);
        let words: Vec<&str> = got.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(words, vec!["ash", "birch", "cedar"]);
        assert!(got.iter().all(|w| w.count == 1));
    }

    #[test]
    fn limit_truncates_to_the_top_n() {
        let text = "storm storm storm rain rain wind fog";
        // Top 2 by count: storm (3), rain (2). The single-occurrence words are cut.
        let got = word_frequencies(text, 2, |_| false);
        assert_eq!(
            got,
            vec![
                WordCount {
                    word: "storm".into(),
                    count: 3
                },
                WordCount {
                    word: "rain".into(),
                    count: 2
                },
            ]
        );
        // A literal limit of 0 yields nothing (callers clamp before calling).
        assert!(word_frequencies(text, 0, |_| false).is_empty());
    }

    #[test]
    fn single_char_tokens_are_dropped() {
        // "a" and "I" are single-char; "ok" survives. Stopword set is empty here,
        // so the single-char filter alone must remove them.
        let text = "a I ok ok";
        let got = word_frequencies(text, 10, |_| false);
        assert_eq!(
            got,
            vec![WordCount {
                word: "ok".into(),
                count: 2
            }]
        );
    }

    #[test]
    fn empty_text_yields_empty() {
        assert!(word_frequencies("", 50, |_| false).is_empty());
        assert!(word_frequencies("   \n\t  ", 50, |_| false).is_empty());
    }

    #[test]
    fn pre_tokenized_path_matches_the_str_path() {
        // A caller with pre-tokenized words must get an identical ranking.
        let words = ["storm", "storm", "rain"].map(String::from);
        let got = word_frequencies_from_words(words, 10, |_| false);
        assert_eq!(
            got,
            vec![
                WordCount {
                    word: "storm".into(),
                    count: 2
                },
                WordCount {
                    word: "rain".into(),
                    count: 1
                },
            ]
        );
    }
}
