/// High-accuracy text analysis primitives that replace fragile Python regex
/// implementations.
use regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;
use unicode_segmentation::UnicodeSegmentation;

use crate::craft::lexicon::{WordMatcher, is_word_char};

// ---------------------------------------------------------------------------
// Static initializers
// ---------------------------------------------------------------------------

fn common_overrides() -> &'static HashMap<&'static str, usize> {
    static INST: OnceLock<HashMap<&str, usize>> = OnceLock::new();
    INST.get_or_init(|| {
        let mut m = HashMap::new();
        for w in [
            "the", "a", "an", "and", "but", "or", "nor", "for", "yet", "so", "is", "am", "are",
            "was", "were", "be", "been", "do", "does", "did", "has", "have", "had", "may", "might",
            "must", "shall", "should", "will", "would", "can", "could", "i", "me", "my", "we",
            "us", "our", "you", "your", "he", "him", "his", "she", "her", "it", "its", "they",
            "them", "their", "this", "that", "these", "those", "who", "whom", "whose", "which",
            "what", "when", "where", "why", "how", "all", "each", "both", "few", "more", "most",
            "some", "no", "not", "one", "two", "just", "if", "then", "than", "too", "as", "at",
            "by", "in", "of", "on", "to", "up", "with", "from", "out", "off", "own", "here",
            "there", "once", "world", "through", "while", "down", "still", "same", "right", "well",
            "back", "much", "great", "old", "new", "first", "last", "long", "big", "high", "small",
            "large", "next", "good", "said", "made", "find", "found", "thought", "think", "know",
            "knew", "known", "see", "saw", "seen", "come", "came", "go", "went", "gone", "take",
            "took", "make", "give", "gave", "tell", "told", "work", "call", "try", "ask", "need",
            "feel", "leave", "left", "put", "mean", "keep", "let", "start", "seem", "help", "show",
            "turn", "move", "live", "point", "change", "play", "close", "night", "real", "part",
            "life", "sure", "since", "thing", "place", "case", "week", "head", "hand", "house",
            "side", "state", "group", "war", "name", "school", "child", "eye", "end", "time",
            "line", "home", "man", "men", "day", "way", "days", "years", "like", "use", "write",
            "read", "run", "set", "got", "get", "done", "says", "want", "look", "say",
        ] {
            m.insert(w, 1);
        }
        for w in [
            "about",
            "after",
            "again",
            "also",
            "before",
            "because",
            "between",
            "children",
            "something",
            "sometimes",
            "nothing",
            "into",
            "very",
            "over",
            "many",
            "even",
            "never",
            "only",
            "other",
            "people",
            "under",
            "any",
            "water",
            "upon",
            "being",
            "woman",
            "women",
            "country",
            "money",
            "himself",
            "themselves",
            "enough",
            "power",
            "number",
            "problem",
            "second",
            "open",
            "during",
            "story",
            "often",
            "human",
            "given",
            "order",
            "early",
            "rather",
            "either",
            "neither",
            "whether",
            "toward",
            "public",
            "program",
            "system",
            "above",
            "itself",
            "behind",
            "morning",
            "without",
            "table",
            "until",
            "among",
            "across",
            "today",
            "someone",
            "almost",
            "always",
            "better",
            "further",
            "father",
            "mother",
            "brother",
            "sister",
            "picture",
            "city",
            "minute",
            "question",
            "able",
            "body",
            "against",
            "became",
            "become",
            "beyond",
            "certain",
            "figure",
            "happen",
            "hundred",
            "husband",
            "island",
            "letter",
            "listen",
            "million",
            "moment",
            "music",
            "nature",
            "office",
            "party",
            "person",
            "perhaps",
            "piece",
            "present",
            "river",
            "simple",
            "sorry",
            "student",
            "study",
            "teacher",
            "thousand",
            "trouble",
            "written",
            "ago",
            "hello",
            "nation",
            "action",
            "reason",
            "season",
            "service",
            "knowledge",
        ] {
            m.insert(w, 2);
        }
        for w in [
            "every",
            "another",
            "beautiful",
            "different",
            "important",
            "however",
            "already",
            "family",
            "company",
            "several",
            "possible",
            "probably",
            "history",
            "together",
            "government",
            "remember",
            "hospital",
            "continue",
            "general",
            "animal",
            "anything",
            "actually",
            "example",
            "idea",
            "imagine",
            "area",
            "everyone",
            "everything",
            "develop",
            "beginning",
            "attention",
            "yesterday",
            "energy",
            "industry",
            "wonderful",
            "following",
            "anyway",
            "character",
            "dangerous",
            "difficult",
            "direction",
            "disappear",
            "discover",
            "election",
            "evidence",
            "exercise",
            "interest",
            "library",
            "magazine",
            "material",
            "minister",
            "newspaper",
            "opinion",
            "organize",
            "personal",
            "position",
            "president",
            "prisoner",
            "quality",
            "radio",
            "realize",
            "separate",
            "tomorrow",
            "typical",
            "usual",
            "violence",
            "victory",
            "computer",
            "consider",
            "comfortable",
        ] {
            m.insert(w, 3);
        }
        for w in [
            "education",
            "information",
            "communication",
            "understanding",
            "experience",
            "especially",
            "relationship",
            "opportunity",
            "investigation",
            "unfortunately",
            "individual",
            "responsibility",
            "conversation",
            "imagination",
            "celebration",
            "unnecessary",
            "extraordinary",
            "independence",
            "entertainment",
        ] {
            m.insert(w, 4);
        }
        for w in [
            "information",
            "especially",
            "individual",
            "experience",
            "opportunity",
            "particular",
            "community",
            "relationship",
            "technology",
            "understanding",
            "absolutely",
            "university",
            "dictionary",
            "literature",
            "television",
            "appreciate",
            "immediately",
            "necessary",
            "conversation",
            "everybody",
            "generation",
            "independent",
            "population",
            "celebration",
            "situation",
            "communication",
        ] {
            m.insert(w, 4);
        }
        for w in ["responsibility", "international", "organization"] {
            m.insert(w, 5);
        }
        m
    })
}

/// Word segments of `text` with their byte offsets.
///
/// Segment rule: `text` is cut at UAX #29 word boundaries
/// (`split_word_bound_indices`) and a segment is kept iff it contains at least
/// one alphanumeric character (`char::is_alphanumeric`). So "don't", "o'clock",
/// "rock'n'roll", "3.14", "1,000", "U.S.A", "snake_case" and a decomposed
/// "cafe\u{301}" are single words; whitespace, punctuation runs, lone quotation
/// marks and symbols are dropped; a hyphen or dash splits ("well-known" gives
/// "well", "known"); each CJK ideograph is its own word. A kept word never
/// starts or ends with a quotation mark and never contains whitespace.
pub(crate) fn unicode_words(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.split_word_bound_indices()
        .filter(|(_, segment)| segment.chars().any(char::is_alphanumeric))
}

/// Maximal runs of word characters ([`is_word_char`]) with byte offsets: the
/// regex `\w+`. Unlike [`unicode_words`], apostrophes, periods and commas always
/// split ("don't" gives "don", "t"; "3.14" gives "3", "14"), so a run is exactly
/// the text a `\b`-anchored pattern would see.
pub(crate) fn word_char_runs(text: &str) -> impl Iterator<Item = (usize, &str)> {
    let mut chars = text.char_indices().peekable();
    std::iter::from_fn(move || {
        let (start, _) = chars.by_ref().find(|&(_, c)| is_word_char(c))?;
        let mut end = text.len();
        while let Some(&(i, c)) = chars.peek() {
            if !is_word_char(c) {
                end = i;
                break;
            }
            chars.next();
        }
        Some((start, &text[start..end]))
    })
}

/// Text inside each quotation: straight double quotes, curly double quotes
/// and curly single quotes, scanned left to right without overlap and closed
/// by the nearest matching mark (newlines included). An opening curly double
/// quote with no closing mark runs to the end of the text (multi-paragraph
/// dialogue); any other unpaired opening mark is skipped.
fn quoted_spans(text: &str) -> impl Iterator<Item = &str> {
    let mut at = 0;
    std::iter::from_fn(move || {
        while let Some(open) = text[at..].chars().next() {
            let close = match open {
                '\u{201c}' => '\u{201d}',
                '\u{2018}' => '\u{2019}',
                '"' => '"',
                _ => {
                    at += open.len_utf8();
                    continue;
                }
            };
            let body = at + open.len_utf8();
            at = body;
            match text[body..].find(close) {
                Some(len) => {
                    at = body + len + close.len_utf8();
                    return Some(&text[body..body + len]);
                }
                None if open == '\u{201c}' => {
                    at = text.len();
                    return Some(&text[body..]);
                }
                None => {}
            }
        }
        None
    })
}

/// Passive-voice regex: a `be`-form auxiliary followed by an `-ed`/`-en` participle.
pub fn passive_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:was|were|been|being|is|are)\s+\w+(?:ed|en)\b")
            .expect("invalid passive regex")
    })
}

// ---------------------------------------------------------------------------
// count_syllables
// ---------------------------------------------------------------------------

/// High-accuracy syllable counter. Uses a static override map for common
/// English words and handles silent-e patterns and multi-vowel exceptions.
pub fn count_syllables(word: &str) -> usize {
    // Only ASCII letters take part; everything else (digits, punctuation,
    // non-Latin letters) is dropped after Unicode lowercasing, which is how
    // characters such as U+212A KELVIN SIGN still contribute their ASCII form.
    let mut alpha = String::with_capacity(word.len());
    for c in word.chars() {
        if c.is_ascii() {
            if c.is_ascii_alphabetic() {
                alpha.push(c.to_ascii_lowercase());
            }
        } else {
            alpha.extend(c.to_lowercase().filter(char::is_ascii_alphabetic));
        }
    }
    if alpha.is_empty() {
        return 0;
    }

    if let Some(&count) = common_overrides().get(alpha.as_str()) {
        return count;
    }

    // Multi-letter sequence adjustments: vowel pairs the group counter sees as
    // one group but English pronounces as two syllables ("official", "hideous",
    // "curious"), minus the "-tia"/"-ial"/"-tious" spellings where it does not.
    let mut adjustment: i32 = 0;
    if alpha.contains("ia") && !alpha.contains("tia") && !alpha.contains("ial") {
        adjustment += 1;
    }
    if alpha.contains("eous") {
        adjustment += 1;
    }
    if alpha.contains("ious") && !alpha.contains("tious") {
        adjustment += 1;
    }

    // A trailing "e" is silent unless the word ends in "-le".
    let bytes = alpha.as_bytes();
    let mut len = bytes.len();
    if len > 2 && bytes[len - 1] == b'e' && !alpha.ends_with("le") {
        len -= 1;
    }

    // Count vowel groups; "y" is a vowel except in first position.
    let mut count: i32 = 0;
    let mut in_vowel = false;
    for (i, &b) in bytes[..len].iter().enumerate() {
        let is_vowel = matches!(b, b'a' | b'e' | b'i' | b'o' | b'u') || (b == b'y' && i > 0);
        if is_vowel {
            if !in_vowel {
                count += 1;
                in_vowel = true;
            }
        } else {
            in_vowel = false;
        }
    }

    (count + adjustment).max(1) as usize
}

// ---------------------------------------------------------------------------
// count_sentences
// ---------------------------------------------------------------------------

/// Accurate sentence counter. Handles abbreviations, decimal numbers,
/// ellipses, Unicode ellipsis, and combined punctuation like "?!".
pub fn count_sentences(text: &str) -> usize {
    crate::substrate::utils::sentence_spans(text).len()
}

// ---------------------------------------------------------------------------
// estimate_dialogue_tokens
// ---------------------------------------------------------------------------

/// Count words inside quoted text. Supports straight double quotes, curly
/// double quotes, curly single quotes, and multi-paragraph dialogue.
pub fn estimate_dialogue_tokens(text: &str) -> usize {
    quoted_spans(text)
        .map(|quote| unicode_words(quote).count())
        .sum()
}

// ---------------------------------------------------------------------------
// tokenize_words
// ---------------------------------------------------------------------------

/// Fast word tokenizer. Splits on whitespace/punctuation, preserves internal
/// apostrophes, lowercases all tokens, filters empties.
pub fn tokenize_words(text: &str) -> Vec<String> {
    word_slices(text).map(normalize_word).collect()
}

/// The words of [`unicode_words`] (letters, digits, underscore, internal
/// apostrophes and number punctuation). `craft::readability::word_slices`
/// deliberately differs: it keeps only alphabetic runs because syllable
/// counting has no use for digits.
pub(crate) fn word_slices(text: &str) -> impl Iterator<Item = &str> {
    unicode_words(text).map(|(_, word)| word)
}

/// The prefix of `text` holding approximately its first `n` whitespace-delimited
/// words (the cut lands on the `n`th whitespace character).
pub(crate) fn first_words(text: &str, n: usize) -> &str {
    let mut count = 0;
    let mut end = text.len();
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            count += 1;
            if count >= n {
                end = i;
                break;
            }
        }
    }
    &text[..end]
}

pub(crate) fn normalize_word(word: &str) -> String {
    let lower = word.trim_matches(['\'', '‘', '’']).to_lowercase();
    if lower.contains(['‘', '’']) {
        lower.replace(['‘', '’'], "'")
    } else {
        lower
    }
}

pub fn stopwords() -> &'static std::collections::HashSet<&'static str> {
    static INST: OnceLock<std::collections::HashSet<&'static str>> = OnceLock::new();
    INST.get_or_init(|| {
        [
            "a",
            "about",
            "above",
            "after",
            "again",
            "against",
            "all",
            "am",
            "an",
            "and",
            "any",
            "are",
            "aren't",
            "as",
            "at",
            "be",
            "because",
            "been",
            "before",
            "being",
            "below",
            "between",
            "both",
            "but",
            "by",
            "can",
            "can't",
            "cannot",
            "could",
            "couldn't",
            "did",
            "didn't",
            "do",
            "does",
            "doesn't",
            "doing",
            "don't",
            "down",
            "during",
            "each",
            "few",
            "for",
            "from",
            "further",
            "get",
            "got",
            "had",
            "hadn't",
            "has",
            "hasn't",
            "have",
            "haven't",
            "having",
            "he",
            "he'd",
            "he'll",
            "he's",
            "her",
            "here",
            "here's",
            "hers",
            "herself",
            "him",
            "himself",
            "his",
            "how",
            "how's",
            "i",
            "i'd",
            "i'll",
            "i'm",
            "i've",
            "if",
            "in",
            "into",
            "is",
            "isn't",
            "it",
            "it's",
            "its",
            "itself",
            "just",
            "let's",
            "me",
            "more",
            "most",
            "mustn't",
            "my",
            "myself",
            "no",
            "nor",
            "not",
            "of",
            "off",
            "on",
            "once",
            "only",
            "or",
            "other",
            "ought",
            "our",
            "ours",
            "ourselves",
            "out",
            "over",
            "own",
            "same",
            "shan't",
            "she",
            "she'd",
            "she'll",
            "she's",
            "should",
            "shouldn't",
            "so",
            "some",
            "such",
            "than",
            "that",
            "that's",
            "the",
            "their",
            "theirs",
            "them",
            "themselves",
            "then",
            "there",
            "there's",
            "these",
            "they",
            "they'd",
            "they'll",
            "they're",
            "they've",
            "this",
            "those",
            "through",
            "to",
            "too",
            "under",
            "until",
            "up",
            "very",
            "was",
            "wasn't",
            "we",
            "we'd",
            "we'll",
            "we're",
            "we've",
            "were",
            "weren't",
            "what",
            "what's",
            "when",
            "when's",
            "where",
            "where's",
            "which",
            "while",
            "who",
            "who's",
            "whom",
            "why",
            "why's",
            "will",
            "with",
            "won't",
            "would",
            "wouldn't",
            "you",
            "you'd",
            "you'll",
            "you're",
            "you've",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "also",
            "back",
            "come",
            "even",
            "go",
            "going",
            "gone",
            "good",
            "got",
            "know",
            "like",
            "look",
            "made",
            "make",
            "much",
            "new",
            "now",
            "one",
            "really",
            "right",
            "said",
            "say",
            "see",
            "still",
            "take",
            "tell",
            "thing",
            "think",
            "time",
            "two",
            "us",
            "use",
            "want",
            "way",
            "well",
            "went",
        ]
        .into_iter()
        .collect()
    })
}

pub fn content_tokens(
    text: &str,
    stopwords: &std::collections::HashSet<&'static str>,
) -> Vec<String> {
    tokenize_words(text)
        .into_iter()
        .filter(|w| w.len() >= 3 && !stopwords.contains(w.as_str()))
        .collect()
}

// ---------------------------------------------------------------------------
// Prose quality: show-don't-tell & word-echo detection
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ProseIssue {
    pub scene_id: usize,
    pub issue_type: String,
    pub snippet: String,
    pub suggestion: String,
}

// -- Static regex helpers for show-don't-tell --

fn trigger_verbs() -> &'static WordMatcher {
    static RE: OnceLock<WordMatcher> = OnceLock::new();
    RE.get_or_init(|| WordMatcher::new(&["was", "felt", "seemed", "appeared", "looked"]))
}

struct TellingEntry {
    word: &'static str,
    category: &'static str,
}

struct CompiledTelling {
    word: &'static str,
    category: &'static str,
    re: Regex,
}

fn compiled_telling_words() -> &'static [CompiledTelling] {
    static INST: OnceLock<Vec<CompiledTelling>> = OnceLock::new();
    INST.get_or_init(|| {
        telling_words()
            .iter()
            .map(|e| CompiledTelling {
                word: e.word,
                category: e.category,
                re: word_boundary_regex(e.word),
            })
            .collect()
    })
}

fn telling_words() -> &'static [TellingEntry] {
    static INST: OnceLock<Vec<TellingEntry>> = OnceLock::new();
    INST.get_or_init(|| {
        vec![
            TellingEntry {
                word: "angry",
                category: "emotion",
            },
            TellingEntry {
                word: "sad",
                category: "emotion",
            },
            TellingEntry {
                word: "happy",
                category: "emotion",
            },
            TellingEntry {
                word: "afraid",
                category: "emotion",
            },
            TellingEntry {
                word: "nervous",
                category: "emotion",
            },
            TellingEntry {
                word: "excited",
                category: "emotion",
            },
            TellingEntry {
                word: "beautiful",
                category: "filter",
            },
            TellingEntry {
                word: "ugly",
                category: "filter",
            },
            TellingEntry {
                word: "interesting",
                category: "filter",
            },
            TellingEntry {
                word: "boring",
                category: "filter",
            },
            TellingEntry {
                word: "amazing",
                category: "filter",
            },
            TellingEntry {
                word: "terrible",
                category: "filter",
            },
            TellingEntry {
                word: "wonderful",
                category: "filter",
            },
            TellingEntry {
                word: "horrible",
                category: "filter",
            },
        ]
    })
}

/// Case-insensitive whole-word regex matching the literal `word` (escaped). The
/// one home for the `(?i)\b{word}\b` idiom used across analyzers.
pub fn word_boundary_regex(word: &str) -> Regex {
    Regex::new(&format!(r"(?i)\b{}\b", regex::escape(word))).expect("invalid word-boundary regex")
}

struct AdvancedPattern {
    regex: Regex,
    ptype: &'static str,
    suggestion: &'static str,
}

fn advanced_patterns() -> &'static [AdvancedPattern] {
    static INST: OnceLock<Vec<AdvancedPattern>> = OnceLock::new();
    INST.get_or_init(|| {
        vec![
            AdvancedPattern {
                regex: Regex::new(r"(?i)\b(he|she|they|i)('s|'d)?\s+(had\s+)?(felt|knew|realized|thought)\s+that\b").expect("regex"),
                ptype: "narrative_distance",
                suggestion: "The narrator is explaining the character's perception. Let the reader infer it from action/dialogue.",
            },
            AdvancedPattern {
                regex: Regex::new(r"(?i)\bit\s+(was|became)\s+(clear|obvious|apparent)\s+that\b").expect("regex"),
                ptype: "editorial",
                suggestion: "The narrator is telling the reader what to conclude. Show the evidence and let them draw the conclusion.",
            },
            AdvancedPattern {
                regex: Regex::new(r"(?i)\b(he|she|they|i)\s+(could|managed\s+to)\s+(see|hear|feel|smell)\b").expect("regex"),
                ptype: "filter_word",
                suggestion: "Remove the filter and describe directly what is perceived. 'She could see the flames' -> 'Flames licked the curtains.'",
            },
            AdvancedPattern {
                regex: Regex::new(r"(?i)\b(he|she|they|i)\s+(started|began)\s+to\b").expect("regex"),
                ptype: "diluted_action",
                suggestion: "Remove the diluter and use the direct verb. 'She started to run' -> 'She ran.'",
            },
        ]
    })
}

/// The largest char boundary `<= i` (clamped to `s.len()`). Use before slicing with
/// a byte offset derived from arithmetic (windowing, `saturating_sub`), which can land
/// inside a multibyte char on real text and panic.
pub(crate) fn floor_char_boundary(s: &str, i: usize) -> usize {
    let mut i = i.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// The smallest char boundary `>= i` (clamped to `s.len()`). Companion to
/// [`floor_char_boundary`] for the upper bound of a slice.
pub(crate) fn ceil_char_boundary(s: &str, i: usize) -> usize {
    let mut i = i.min(s.len());
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// `s` unchanged when it fits in `max_bytes`, otherwise cut at the last char
/// boundary within `max_bytes` with `...` appended.
pub(crate) fn truncate_snippet(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        s.to_string()
    } else {
        format!("{}...", &s[..floor_char_boundary(s, max_bytes)])
    }
}

/// Split `text` into paragraphs, the canonical (§6.1) definition shared across
/// the core. A paragraph break is one or more blank lines, where "blank" means a
/// line that is empty or whitespace-only — so `"\n\n"`, `"\r\n\r\n"`, and
/// `"\n   \n"` all separate. Empty paragraphs (runs of blank lines) are dropped;
/// each returned paragraph is trimmed.
pub fn split_paragraphs(text: &str) -> Vec<&str> {
    let mut paragraphs = Vec::new();
    let mut start: Option<usize> = None;
    for line in text.split_inclusive('\n') {
        let offset = line.as_ptr() as usize - text.as_ptr() as usize;
        if line.trim().is_empty() {
            if let Some(s) = start.take() {
                paragraphs.push(text[s..offset].trim());
            }
        } else if start.is_none() {
            start = Some(offset);
        }
    }
    if let Some(s) = start {
        paragraphs.push(text[s..].trim());
    }
    paragraphs
}

pub fn split_sentences(text: &str) -> Vec<&str> {
    // Abbreviation-aware boundaries from the shared primitive (not the fragile
    // `[.!?]\s+` regex), returned as borrowed slices.
    crate::substrate::utils::sentence_spans(text)
        .into_iter()
        .map(|(s, e)| &text[s..e])
        .collect()
}

/// Split `text` into scene blocks (paragraphs separated by blank lines), each
/// paired with the UTF-16 code-unit offsets of its trimmed content in `text`.
/// UTF-16 is the offset unit editors use for edit ranges, so a returned span
/// can be navigated to directly.
pub fn paragraph_spans(text: &str) -> Vec<(String, u32, u32)> {
    let paragraphs = split_paragraphs(text);
    let byte_spans: Vec<(usize, usize)> = paragraphs
        .iter()
        .map(|piece| {
            let start = piece.as_ptr() as usize - text.as_ptr() as usize;
            (start, start + piece.len())
        })
        .collect();
    let mut texts: Vec<String> = paragraphs.into_iter().map(str::to_owned).collect();
    // 2. Map byte offsets to UTF-16 offsets in one forward pass (spans are
    //    ascending and non-overlapping).
    let mut result = Vec::with_capacity(byte_spans.len());
    let mut chars = text.char_indices().peekable();
    let mut utf16: u32 = 0;
    for (i, (start_byte, end_byte)) in byte_spans.iter().enumerate() {
        while let Some(&(b, ch)) = chars.peek() {
            if b >= *start_byte {
                break;
            }
            utf16 += ch.len_utf16() as u32;
            chars.next();
        }
        let start16 = utf16;
        while let Some(&(b, ch)) = chars.peek() {
            if b >= *end_byte {
                break;
            }
            utf16 += ch.len_utf16() as u32;
            chars.next();
        }
        result.push((std::mem::take(&mut texts[i]), start16, utf16));
    }
    result
}

pub fn detect_show_dont_tell(scene_text: &str, scene_id: usize) -> Vec<ProseIssue> {
    const MAX_INPUT: usize = 200_000;
    const MAX_ISSUES: usize = 15;

    if scene_text.is_empty() {
        return Vec::new();
    }

    let text = if scene_text.len() > MAX_INPUT {
        &scene_text[..floor_char_boundary(scene_text, MAX_INPUT)]
    } else {
        scene_text
    };
    let text = text.trim();
    let sentences = split_sentences(text);
    let mut results = Vec::new();

    // Basic detection: trigger verb + telling word
    let trigger_re = trigger_verbs();
    let words = compiled_telling_words();

    for sent in &sentences {
        if results.len() >= MAX_ISSUES {
            break;
        }
        let lower = sent.to_lowercase();
        if !trigger_re.is_match(&lower) {
            continue;
        }
        for entry in words {
            if results.len() >= MAX_ISSUES {
                break;
            }
            if entry.re.is_match(&lower) {
                let snippet = truncate_snippet(sent, 120);
                let suggestion = match entry.category {
                    "emotion" => format!(
                        "Instead of saying the character '{}', show the physical manifestation (clenched fists, flushed face)",
                        entry.word
                    ),
                    _ => format!(
                        "Instead of labeling something '{}', show specific sensory details that let the reader conclude it",
                        entry.word
                    ),
                };
                results.push(ProseIssue {
                    scene_id,
                    issue_type: "show_dont_tell".to_string(),
                    snippet,
                    suggestion,
                });
                break; // one flag per sentence
            }
        }
    }

    // Advanced detection
    let patterns = advanced_patterns();
    for sent in &sentences {
        if results.len() >= MAX_ISSUES {
            break;
        }
        for pat in patterns {
            if pat.regex.is_match(sent) {
                results.push(ProseIssue {
                    scene_id,
                    issue_type: format!("show_dont_tell_{}", pat.ptype),
                    snippet: truncate_snippet(sent, 120),
                    suggestion: pat.suggestion.to_string(),
                });
                break; // one flag per sentence
            }
        }
    }

    results.truncate(MAX_ISSUES);
    results
}

// -- Word echo detection --

/// Common narrative words whose repetition is not an echo worth flagging; a
/// narrower, curated list than `stopwords`, which would suppress real echoes
/// such as "thing" or "really".
fn echo_stopwords() -> &'static std::collections::HashSet<&'static str> {
    static INST: OnceLock<std::collections::HashSet<&str>> = OnceLock::new();
    INST.get_or_init(|| {
        [
            "the", "and", "that", "with", "from", "have", "been", "were", "this", "they", "their",
            "which", "would", "could", "should", "about", "there", "other", "after", "before",
            "these", "those", "then", "than", "them", "what", "when", "where", "while", "being",
            "does", "done", "each", "every", "into", "just", "like", "make", "made", "many",
            "more", "most", "much", "must", "never", "only", "over", "said", "same", "some",
            "such", "take", "tell", "also", "back", "come", "even", "first", "give", "good",
            "great", "hand", "here", "high", "know", "last", "long", "look", "name", "next",
            "part", "place", "point", "right", "small", "still", "think", "through", "turn",
            "under", "upon", "very", "want", "well", "will", "work", "world", "year", "your",
        ]
        .into_iter()
        .collect()
    })
}

pub fn detect_word_echoes(scene_text: &str, scene_id: usize) -> Vec<ProseIssue> {
    const WINDOW_SIZE: usize = 100;
    const STEP: usize = 50;
    const MIN_WORD_LEN: usize = 5;
    const MIN_REPEATS: usize = 3;
    const MAX_ISSUES: usize = 10;

    // Echo detection runs on lowercased text and compares ASCII letter runs
    // only, so "well-known" contributes "well" and "known" and "don't"
    // contributes "don" and "t"; this is intentionally narrower than
    // [`word_slices`].
    let lower = scene_text.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_lowercase())
        .filter(|word| !word.is_empty())
        .collect();

    if words.is_empty() {
        return Vec::new();
    }

    let windows: Vec<&[&str]> = if words.len() < WINDOW_SIZE {
        vec![&words[..]]
    } else {
        (0..=(words.len() - WINDOW_SIZE))
            .step_by(STEP)
            .map(|i| &words[i..i + WINDOW_SIZE])
            .collect()
    };

    let stopwords = echo_stopwords();
    let mut seen_words = std::collections::HashSet::new();
    let mut results = Vec::new();

    for window in &windows {
        if results.len() >= MAX_ISSUES {
            break;
        }
        let mut freq: HashMap<&str, usize> = HashMap::new();
        for &w in *window {
            if w.len() >= MIN_WORD_LEN && !stopwords.contains(w) {
                *freq.entry(w).or_insert(0) += 1;
            }
        }
        let mut echoes: Vec<(&str, usize)> = freq
            .into_iter()
            .filter(|&(w, count)| count >= MIN_REPEATS && !seen_words.contains(w))
            .collect();
        echoes.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        for (w, count) in echoes {
            if results.len() >= MAX_ISSUES {
                break;
            }
            seen_words.insert(w.to_string());
            let snippet_text = window
                .iter()
                .take(30)
                .copied()
                .collect::<Vec<_>>()
                .join(" ")
                + "...";
            results.push(ProseIssue {
                scene_id,
                issue_type: "word_echo".to_string(),
                snippet: snippet_text,
                suggestion: format!(
                    "The word '{}' appears {} times in a short span. Consider using synonyms or restructuring.",
                    w, count
                ),
            });
        }
    }

    results
}

// ---------------------------------------------------------------------------
// UTF-8 <-> UTF-16 offset arithmetic
// ---------------------------------------------------------------------------

/// UTF-16 code-unit length of `s`. Native editors (NSTextView, WinUI, Compose,
/// browser `contenteditable`) index text in UTF-16 code units, not bytes.
pub fn utf16_len(s: &str) -> u32 {
    s.chars().map(|c| c.len_utf16() as u32).sum()
}

/// Convert a UTF-8 byte offset within `s` into a UTF-16 code-unit offset.
///
/// `byte_offset` must fall on a `char` boundary within `s` (the byte index of a
/// char start, or `s.len()`); callers pass match boundaries from `&str` search,
/// which are always on boundaries. Returns `None` otherwise.
pub fn byte_to_utf16_offset(s: &str, byte_offset: usize) -> Option<u32> {
    if !s.is_char_boundary(byte_offset) {
        return None;
    }
    let mut u16_count: u32 = 0;
    for (idx, ch) in s.char_indices() {
        if idx == byte_offset {
            return Some(u16_count);
        }
        if idx > byte_offset {
            return None;
        }
        u16_count += ch.len_utf16() as u32;
    }
    Some(u16_count)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf16_offsets_require_utf8_boundaries() {
        let cases: &[(&str, &[Option<u32>])] = &[
            ("", &[Some(0)]),
            ("a", &[Some(0), Some(1)]),
            ("ي", &[Some(0), None, Some(1)]),
            ("雪", &[Some(0), None, None, Some(1)]),
            ("🖋", &[Some(0), None, None, None, Some(2)]),
            (
                "a🖋é",
                &[Some(0), Some(1), None, None, None, Some(3), None, Some(4)],
            ),
        ];
        for &(text, offsets) in cases {
            for (offset, &expected) in offsets.iter().enumerate() {
                assert_eq!(
                    byte_to_utf16_offset(text, offset),
                    expected,
                    "{text:?} at {offset}"
                );
            }
            assert_eq!(byte_to_utf16_offset(text, text.len() + 1), None);
            assert_eq!(byte_to_utf16_offset(text, usize::MAX), None);
        }
    }

    #[test]
    fn split_paragraphs_handles_blank_line_variants() {
        // Plain "\n\n".
        assert_eq!(split_paragraphs("one\n\ntwo"), vec!["one", "two"]);
        // CRLF blank line.
        assert_eq!(split_paragraphs("one\r\n\r\ntwo"), vec!["one", "two"]);
        // Whitespace-only blank line separates.
        assert_eq!(split_paragraphs("one\n   \ntwo"), vec!["one", "two"]);
        // Runs of blank lines collapse to a single break, no empty paragraphs.
        assert_eq!(split_paragraphs("one\n\n\n\ntwo"), vec!["one", "two"]);
        // Single newlines do not split.
        assert_eq!(split_paragraphs("a\nb\nc"), vec!["a\nb\nc"]);
        // Leading/trailing blank lines are dropped.
        assert_eq!(split_paragraphs("\n\none\n\n"), vec!["one"]);
        // Empty / whitespace-only input yields no paragraphs.
        assert!(split_paragraphs("").is_empty());
        assert!(split_paragraphs("  \n \n").is_empty());
    }

    #[test]
    fn split_sentences_is_abbreviation_aware() {
        // Was the fragile `[.!?]\s+` regex; now routes through the shared
        // abbreviation-aware segmenter and returns borrowed slices.
        let s = split_sentences("Dr. Smith left at 3.14 p.m. He never returned.");
        assert_eq!(s.len(), 2, "got {s:?}");
        assert!(s[0].starts_with("Dr. Smith"));
        assert_eq!(s[1], "He never returned.");
    }

    // -- count_syllables --

    #[test]
    fn test_syllables_monosyllabic() {
        assert_eq!(count_syllables("cat"), 1);
        assert_eq!(count_syllables("the"), 1);
        assert_eq!(count_syllables("through"), 1);
        assert_eq!(count_syllables("world"), 1);
    }

    #[test]
    fn test_syllables_silent_e() {
        assert_eq!(count_syllables("make"), 1);
        assert_eq!(count_syllables("time"), 1);
        assert_eq!(count_syllables("use"), 1);
    }

    #[test]
    fn test_syllables_le_ending() {
        assert_eq!(count_syllables("table"), 2);
        assert_eq!(count_syllables("simple"), 2);
    }

    #[test]
    fn test_syllables_bisyllabic() {
        assert_eq!(count_syllables("people"), 2);
        assert_eq!(count_syllables("water"), 2);
        assert_eq!(count_syllables("after"), 2);
        assert_eq!(count_syllables("hello"), 2);
        assert_eq!(count_syllables("nation"), 2);
    }

    #[test]
    fn test_syllables_trisyllabic() {
        assert_eq!(count_syllables("every"), 3);
        assert_eq!(count_syllables("beautiful"), 3);
        assert_eq!(count_syllables("different"), 3);
    }

    #[test]
    fn test_syllables_polysyllabic() {
        assert_eq!(count_syllables("information"), 4);
        assert_eq!(count_syllables("responsibility"), 5);
        assert_eq!(count_syllables("communication"), 4);
    }

    #[test]
    fn test_syllables_empty() {
        assert_eq!(count_syllables(""), 0);
        assert_eq!(count_syllables("   "), 0);
        assert_eq!(count_syllables("123"), 0);
    }

    #[test]
    fn test_syllables_case_insensitive() {
        assert_eq!(count_syllables("The"), count_syllables("the"));
        assert_eq!(count_syllables("EVERY"), count_syllables("every"));
    }

    #[test]
    fn test_syllables_minimum() {
        assert_eq!(count_syllables("x"), 1);
        assert_eq!(count_syllables("sh"), 1);
    }

    #[test]
    fn test_syllables_basic_compat() {
        assert_eq!(count_syllables("hello"), 2);
        assert_eq!(count_syllables("world"), 1);
        assert_eq!(count_syllables("beautiful"), 3);
        assert_eq!(count_syllables("the"), 1);
        assert_eq!(count_syllables("a"), 1);
    }

    // -- count_sentences --

    #[test]
    fn test_sentences_simple() {
        assert_eq!(count_sentences("Hello world. How are you? Fine!"), 3);
    }

    #[test]
    fn test_sentences_abbreviations() {
        assert_eq!(
            count_sentences("Dr. Smith went to Washington. He arrived."),
            2
        );
        assert_eq!(count_sentences("I met Mr. Jones at the park."), 1);
    }

    #[test]
    fn test_sentences_decimal() {
        assert_eq!(count_sentences("The price is 3.14 dollars."), 1);
    }

    #[test]
    fn test_sentences_ellipsis() {
        assert_eq!(count_sentences("Wait... really? Yes!"), 3);
    }

    #[test]
    fn test_sentences_unicode_ellipsis() {
        assert_eq!(count_sentences("Wait\u{2026} really? Yes!"), 3);
    }

    #[test]
    fn test_sentences_empty() {
        assert_eq!(count_sentences(""), 0);
        assert_eq!(count_sentences("   "), 0);
    }

    #[test]
    fn test_sentences_no_punctuation() {
        assert_eq!(count_sentences("Hello world"), 1);
    }

    #[test]
    fn test_sentences_combined_punctuation() {
        assert_eq!(count_sentences("Really?! Yes."), 2);
    }

    #[test]
    fn test_sentences_eg_ie() {
        assert_eq!(
            count_sentences("Use tools, e.g. a hammer. Also try i.e. screwdrivers."),
            2
        );
    }

    #[test]
    fn test_sentences_exclamation() {
        assert_eq!(count_sentences("Wow! Amazing! Great."), 3);
    }

    #[test]
    fn test_sentences_basic_counts() {
        assert_eq!(count_sentences("One. Two. Three."), 3);
    }

    #[test]
    fn test_sentences_basic_compat() {
        assert_eq!(count_sentences("Hello world. How are you?"), 2);
    }

    // -- estimate_dialogue_tokens --

    #[test]
    fn test_dialogue_straight_quotes() {
        let text = r#"She said "hello world" and left."#;
        assert_eq!(estimate_dialogue_tokens(text), 2);
    }

    #[test]
    fn test_dialogue_curly_double_quotes() {
        let text = "He whispered \u{201c}come here quickly\u{201d} to her.";
        assert_eq!(estimate_dialogue_tokens(text), 3);
    }

    #[test]
    fn test_dialogue_curly_single_quotes() {
        let text = "She said \u{2018}not now\u{2019} firmly.";
        assert_eq!(estimate_dialogue_tokens(text), 2);
    }

    #[test]
    fn test_dialogue_multiple() {
        let text = r#""Hello," she said. "How are you?""#;
        assert_eq!(estimate_dialogue_tokens(text), 4);
    }

    #[test]
    fn test_dialogue_empty() {
        assert_eq!(estimate_dialogue_tokens("No dialogue here."), 0);
    }

    #[test]
    fn test_dialogue_multiline_straight() {
        let text = "\"Unclosed quote across\ntwo lines\" she finished";
        assert_eq!(estimate_dialogue_tokens(text), 5);
    }

    #[test]
    fn test_dialogue_mixed() {
        let text = "He said \"hello there\" and \"goodbye\"\n\u{201c}This is curly\u{201d} she said\n\"Unclosed quote across\ntwo lines\" she finished";
        // "hello there"=2, "goodbye"=1, "This is curly"=3, multiline=5 => 11
        assert_eq!(estimate_dialogue_tokens(text), 11);
    }

    // -- unicode_words / word_char_runs --

    #[test]
    fn word_iterators_map_offsets_back_and_skip_non_words() {
        let text = "“Don’t,” said O'Brien — 3.14 _x_ cafe\u{301}!\n\t‘’ … naïve/you";
        let words: Vec<(usize, &str)> = unicode_words(text).collect();
        let runs: Vec<(usize, &str)> = word_char_runs(text).collect();
        for list in [&words, &runs] {
            assert!(!list.is_empty());
            let mut last_end = 0;
            for &(start, word) in list.iter() {
                assert_eq!(&text[start..start + word.len()], word);
                assert!(start >= last_end, "words are in order and disjoint");
                last_end = start + word.len();
                assert!(!word.is_empty());
                assert!(word.chars().any(char::is_alphanumeric), "{word:?}");
                assert!(!word.chars().any(char::is_whitespace), "{word:?}");
                assert!(!word.starts_with(['“', '‘', '’', '”', '\'']));
                assert!(!word.ends_with(['“', '‘', '’', '”', '\'']));
            }
        }
        assert_eq!(
            words.iter().map(|(_, w)| *w).collect::<Vec<_>>(),
            [
                "Don’t",
                "said",
                "O'Brien",
                "3.14",
                "_x_",
                "cafe\u{301}",
                "naïve",
                "you"
            ]
        );
        assert_eq!(
            runs.iter().map(|(_, w)| *w).collect::<Vec<_>>(),
            [
                "Don",
                "t",
                "said",
                "O",
                "Brien",
                "3",
                "14",
                "_x_",
                "cafe\u{301}",
                "naïve",
                "you"
            ]
        );
    }

    // -- tokenize_words --

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize_words("Hello, World!");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_apostrophes() {
        let tokens = tokenize_words("don't it's O'Brien");
        assert_eq!(tokens, vec!["don't", "it's", "o'brien"]);
    }

    #[test]
    fn test_tokenize_empty() {
        assert!(tokenize_words("").is_empty());
    }

    #[test]
    fn test_tokenize_whitespace_variants() {
        let tokens = tokenize_words("hello\tworld\nnew  line");
        assert_eq!(tokens, vec!["hello", "world", "new", "line"]);
    }

    #[test]
    fn test_tokenize_punctuation_stripped() {
        let tokens = tokenize_words("end. start, mid--dash");
        assert_eq!(tokens, vec!["end", "start", "mid", "dash"]);
    }

    // -- detect_show_dont_tell --

    #[test]
    fn test_show_dont_tell_basic_emotion() {
        let issues = detect_show_dont_tell("She was angry at him.", 0);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].issue_type, "show_dont_tell");
        assert!(issues[0].suggestion.contains("angry"));
    }

    #[test]
    fn test_show_dont_tell_basic_filter() {
        let issues = detect_show_dont_tell("The garden looked beautiful in spring.", 1);
        assert_eq!(issues.len(), 1);
        assert!(issues[0].suggestion.contains("beautiful"));
    }

    #[test]
    fn test_show_dont_tell_no_trigger_verb() {
        let issues = detect_show_dont_tell("He ran across the field.", 0);
        let basic: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == "show_dont_tell")
            .collect();
        assert!(basic.is_empty());
    }

    #[test]
    fn test_show_dont_tell_advanced_narrative_distance() {
        let issues = detect_show_dont_tell("She felt that something was wrong.", 0);
        assert!(
            issues
                .iter()
                .any(|i| i.issue_type == "show_dont_tell_narrative_distance")
        );
    }

    #[test]
    fn test_show_dont_tell_advanced_editorial() {
        let issues = detect_show_dont_tell("It was clear that he would not return.", 0);
        assert!(
            issues
                .iter()
                .any(|i| i.issue_type == "show_dont_tell_editorial")
        );
    }

    #[test]
    fn test_show_dont_tell_advanced_filter_word() {
        let issues = detect_show_dont_tell("She could see the flames rising.", 0);
        assert!(
            issues
                .iter()
                .any(|i| i.issue_type == "show_dont_tell_filter_word")
        );
    }

    #[test]
    fn test_show_dont_tell_advanced_diluted() {
        let issues = detect_show_dont_tell("He started to run toward the exit.", 0);
        assert!(
            issues
                .iter()
                .any(|i| i.issue_type == "show_dont_tell_diluted_action")
        );
    }

    #[test]
    fn test_show_dont_tell_empty() {
        let issues = detect_show_dont_tell("", 0);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_show_dont_tell_cap() {
        // 20 sentences with trigger + telling word; should cap at 15
        let text = (0..20)
            .map(|_| "She was happy.")
            .collect::<Vec<_>>()
            .join(" ");
        let issues = detect_show_dont_tell(&text, 0);
        assert!(issues.len() <= 15);
    }

    // -- detect_word_echoes --

    #[test]
    fn test_word_echoes_detected() {
        // "strange" repeated 4 times in a short window
        let text =
            "The strange door led to a strange room with a strange mirror and a strange light.";
        let issues = detect_word_echoes(text, 0);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].issue_type, "word_echo");
        assert!(issues[0].suggestion.contains("strange"));
    }

    #[test]
    fn test_word_echoes_short_words_ignored() {
        // "the" repeated many times but length < 5, so ignored
        let text = "The the the the the the the the end.";
        let issues = detect_word_echoes(text, 0);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_word_echoes_empty() {
        let issues = detect_word_echoes("", 0);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_word_echoes_stopwords_ignored() {
        // "about" is in stopwords and len >= 5, should be skipped
        let text = (0..10)
            .map(|_| "about something")
            .collect::<Vec<_>>()
            .join(" ");
        let issues = detect_word_echoes(&text, 0);
        // "something" (9 chars, not a stopword) should be flagged, "about" should not
        let about_issues: Vec<_> = issues
            .iter()
            .filter(|i| i.suggestion.contains("'about'"))
            .collect();
        assert!(about_issues.is_empty());
    }

    #[test]
    fn test_word_echoes_cap() {
        // Generate many different repeated words
        let mut text = String::new();
        for i in 0..20 {
            let word = format!("xword{}", i);
            for _ in 0..5 {
                text.push_str(&word);
                text.push(' ');
            }
        }
        let issues = detect_word_echoes(&text, 0);
        assert!(issues.len() <= 10);
    }

    #[test]
    fn test_word_echoes_ordered_by_count_then_word() {
        // Twelve candidates in one window: "zebra" repeats most, the rest tie
        // at three, so the cap keeps "zebra" first and then the first nine
        // alphabetically, dropping "jewel" and "kayak".
        const TIED: [&str; 11] = [
            "apple", "brick", "cedar", "dusty", "ember", "fjord", "glade", "hazel", "ivory",
            "jewel", "kayak",
        ];
        let mut text = String::new();
        for _ in 0..5 {
            text.push_str("zebra ");
        }
        for word in TIED {
            for _ in 0..3 {
                text.push_str(word);
                text.push(' ');
            }
        }
        let flagged: Vec<String> = detect_word_echoes(&text, 0)
            .into_iter()
            .map(|issue| issue.suggestion)
            .collect();
        let mut expected = vec!["'zebra' appears 5 times".to_string()];
        expected.extend(TIED[..9].iter().map(|w| format!("'{w}' appears 3 times")));
        assert_eq!(flagged.len(), expected.len());
        for (got, want) in flagged.iter().zip(&expected) {
            assert!(got.contains(want), "got {got:?}, want {want:?}");
        }
        for _ in 0..20 {
            let again: Vec<String> = detect_word_echoes(&text, 0)
                .into_iter()
                .map(|issue| issue.suggestion)
                .collect();
            assert_eq!(again, flagged);
        }
    }
}
