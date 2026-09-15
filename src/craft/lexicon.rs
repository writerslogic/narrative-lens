//! Whole-word, case-insensitive keyword/phrase matching — the replacement for
//! giant regex alternations like `(?i)\b(a|b|c)\b`. Backed by aho-corasick
//! (build once, e.g. in a `OnceLock`) plus a word-boundary check so matches are
//! whole words, mirroring `\b`. Faster than alternation regex, and the word list
//! stays a plain readable, testable slice.
//!
//! Two matching modes: the default whole-word methods ([`is_match`](WordMatcher::is_match),
//! [`count`](WordMatcher::count)) require a boundary on both sides (`\b(...)\b`);
//! the stem methods ([`is_prefix_match`](WordMatcher::is_prefix_match),
//! [`prefix_count`](WordMatcher::prefix_count)) anchor only the left side
//! (`\b(...)`), so a keyword matches its inflections ("discover" → "discovered")
//! but never fires mid-word ("ally" never matches "really").

use aho_corasick::AhoCorasick;

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn is_word_start(text: &str, start: usize) -> bool {
    text[..start]
        .chars()
        .next_back()
        .is_none_or(|c| !is_word_char(c))
}

fn is_whole_word(text: &str, start: usize, end: usize) -> bool {
    let after_ok = text[end..].chars().next().is_none_or(|c| !is_word_char(c));
    is_word_start(text, start) && after_ok
}

/// A whole-word match: the matched text slice and its byte span in the haystack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WordMatch<'a> {
    pub text: &'a str,
    pub start: usize,
    pub end: usize,
}

/// A set of literal words/phrases matched whole-word and case-insensitively.
pub struct WordMatcher {
    ac: AhoCorasick,
}

impl WordMatcher {
    pub fn new(words: &[&str]) -> Self {
        let ac = AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .build(words)
            .expect("invalid WordMatcher patterns");
        WordMatcher { ac }
    }

    /// True if any word/phrase occurs as a whole word.
    pub fn is_match(&self, text: &str) -> bool {
        self.matches(text).next().is_some()
    }

    /// Count of whole-word occurrences. Whole words cannot overlap, so this
    /// equals the non-overlapping count `(?i)\b(...)\b` would report.
    pub fn count(&self, text: &str) -> usize {
        self.matches(text).count()
    }

    /// Iterator over whole-word matches, yielding the matched slice and span.
    /// Matches are leftmost-first (whole words cannot overlap).
    pub fn find_iter<'a>(&'a self, text: &'a str) -> impl Iterator<Item = WordMatch<'a>> + 'a {
        self.matches(text).map(move |(start, end)| WordMatch {
            text: &text[start..end],
            start,
            end,
        })
    }

    /// First whole-word match, if any.
    pub fn find_first<'a>(&'a self, text: &'a str) -> Option<WordMatch<'a>> {
        self.find_iter(text).next()
    }

    /// True if any word/phrase occurs as a word *stem* — i.e. at the start of a
    /// word, with the right side free to continue (`\b(...)` rather than
    /// `\b(...)\b`). So "govern" matches "government" but not "ungovern", and
    /// "ally" does not match "really". Use this for content-word matching where
    /// inflections should match ("discover" → "discovered") while mid-word
    /// collisions should not; use [`is_match`](Self::is_match) for exact words.
    ///
    /// This is a prefix heuristic, not a real stemmer: it also matches unrelated
    /// words that merely share the prefix ("art" → "article", "artist"). Keep a
    /// minimum-length filter on dynamic keyword lists to bound that risk.
    pub fn is_prefix_match(&self, text: &str) -> bool {
        self.prefix_matches(text).next().is_some()
    }

    /// Count of word-stem occurrences (see [`is_prefix_match`](Self::is_prefix_match)).
    pub fn prefix_count(&self, text: &str) -> usize {
        self.prefix_matches(text).count()
    }

    /// Iterator over word-stem matches, yielding the matched slice and span.
    /// The slice is the keyword as it occurred, not the full surrounding word.
    pub fn find_prefix_iter<'a>(
        &'a self,
        text: &'a str,
    ) -> impl Iterator<Item = WordMatch<'a>> + 'a {
        self.prefix_matches(text)
            .map(move |(start, end)| WordMatch {
                text: &text[start..end],
                start,
                end,
            })
    }

    fn matches<'a>(&'a self, text: &'a str) -> impl Iterator<Item = (usize, usize)> + 'a {
        self.ac
            .find_overlapping_iter(text)
            .map(|m| (m.start(), m.end()))
            .filter(move |&(s, e)| is_whole_word(text, s, e))
    }

    fn prefix_matches<'a>(&'a self, text: &'a str) -> impl Iterator<Item = (usize, usize)> + 'a {
        self.ac
            .find_overlapping_iter(text)
            .map(|m| (m.start(), m.end()))
            .filter(move |&(s, _)| is_word_start(text, s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_whole_words_only() {
        let m = WordMatcher::new(&["said", "asked"]);
        assert!(m.is_match("she said hello"));
        assert!(m.is_match("they ASKED twice"));
        assert!(!m.is_match("unsaid words"));
        assert!(!m.is_match("scatterbrained"));
    }

    #[test]
    fn boundary_matches_regex_b_semantics() {
        let m = WordMatcher::new(&["said"]);
        assert!(m.is_match("said's the word"));
        assert!(m.is_match("\"said\""));
        assert!(!m.is_match("missaid"));
    }

    #[test]
    fn counts_non_overlapping_whole_words() {
        let m = WordMatcher::new(&["ran"]);
        assert_eq!(m.count("she ran and ran again"), 2);
        assert_eq!(m.count("branchran"), 0);
    }

    #[test]
    fn matches_multi_word_phrases() {
        let m = WordMatcher::new(&["turned away", "could have"]);
        assert!(m.is_match("he turned away slowly"));
        assert!(m.is_match("she could have stayed"));
        assert!(!m.is_match("turned awayward"));
    }

    #[test]
    fn find_iter_yields_matched_text_and_span() {
        let m = WordMatcher::new(&["ran", "stood"]);
        let found: Vec<_> = m.find_iter("she ran then stood").collect();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].text, "ran");
        assert_eq!(found[0].start, 4);
        assert_eq!(found[0].end, 7);
        assert_eq!(found[1].text, "stood");
    }

    #[test]
    fn prefix_mode_matches_stems_not_midword() {
        let m = WordMatcher::new(&["discover", "govern"]);
        // Inflections / derived words match (left boundary, free right side).
        assert!(m.is_prefix_match("she discovered the truth"));
        assert!(m.is_prefix_match("the government fell"));
        assert_eq!(m.prefix_count("discover, discovered, discovery"), 3);
        // Mid-word occurrences do not match.
        let ally = WordMatcher::new(&["ally"]);
        assert!(!ally.is_prefix_match("she really finally walked"));
        assert!(ally.is_prefix_match("an ally appeared"));
        // Left boundary still required: "ungovern" is not a stem of "govern".
        assert!(!m.is_prefix_match("an ungoverned land"));
    }

    #[test]
    fn prefix_find_iter_yields_keyword_span() {
        let m = WordMatcher::new(&["fear"]);
        let found: Vec<_> = m.find_prefix_iter("the fearless hero").collect();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].text, "fear");
        assert_eq!(found[0].start, 4);
        assert_eq!(found[0].end, 8);
    }

    #[test]
    fn find_first_is_leftmost_whole_word() {
        let m = WordMatcher::new(&["turned away", "walked out"]);
        let hit = m.find_first("he walked out then turned away").unwrap();
        assert_eq!(hit.text, "walked out");
        assert!(m.find_first("nothing here").is_none());
    }

    #[test]
    fn substring_word_does_not_swallow_later_whole_word() {
        let m = WordMatcher::new(&["he"]);
        assert!(!m.is_match("the"));
        assert!(m.is_match("the he"));
        assert_eq!(m.count("the theatre he"), 1);
    }
}
