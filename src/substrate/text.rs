/// High-accuracy text analysis primitives that replace fragile Python regex
/// implementations.
use aho_corasick::AhoCorasick;
use regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::craft::lexicon::WordMatcher;

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

fn sentence_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(concat!(
            r"(?x)",
            // abbreviations with periods -- match and skip
            r"(?:(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Inc|Ltd|Corp|vs|etc|Vol|Gen|Sgt|Capt|Col|Maj|Rev|Ave|Blvd)\.)",
            r"|",
            // two-part abbreviations
            r"(?:(?:e\.g|i\.e|a\.m|p\.m|U\.S|U\.K)\.)",
            r"|",
            // decimal numbers
            r"(?:\d+\.\d+)",
            r"|",
            // actual sentence boundaries (captured)
            r"([.!?\x{2026}][.!?\x{2026}]*)",
        )).expect("invalid sentence regex")
    })
}

fn dialogue_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(concat!(
            r"(?xs)",
            // curly double quotes (non-greedy to handle multiple on same line)
            r"\x{201c}(.*?)\x{201d}",
            r"|",
            // curly single quotes
            r"\x{2018}(.*?)\x{2019}",
            r"|",
            // straight double quotes (non-greedy, dotall spans newlines)
            r#""(.*?)""#,
            r"|",
            // multi-paragraph: opening curly quote without closing, to end of text
            r"\x{201c}(.*)\z",
        ))
        .expect("invalid dialogue regex")
    })
}

fn word_split_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[^\w']+").expect("invalid word split regex"))
}

/// Passive-voice regex: a `be`-form auxiliary followed by an `-ed`/`-en` participle.
pub fn passive_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:was|were|been|being|is|are)\s+\w+(?:ed|en)\b")
            .expect("invalid passive regex")
    })
}

struct SensoryMatcher {
    automaton: AhoCorasick,
    sense_map: Vec<&'static str>,
}

fn sensory_matcher() -> &'static SensoryMatcher {
    static INST: OnceLock<SensoryMatcher> = OnceLock::new();
    INST.get_or_init(|| {
        let sight: &[&str] = &[
            "see",
            "saw",
            "seen",
            "seeing",
            "look",
            "looked",
            "looking",
            "looks",
            "watch",
            "watched",
            "watching",
            "watches",
            "gaze",
            "gazed",
            "gazing",
            "stare",
            "stared",
            "staring",
            "glance",
            "glanced",
            "glancing",
            "peer",
            "peered",
            "peering",
            "glimpse",
            "glimpsed",
            "squint",
            "squinted",
            "squinting",
            "bright",
            "brighter",
            "brightest",
            "brightly",
            "dark",
            "darker",
            "darkest",
            "darkness",
            "darkly",
            "darken",
            "darkened",
            "shadow",
            "shadows",
            "shadowy",
            "shadowed",
            "shimmer",
            "shimmered",
            "shimmering",
            "glow",
            "glowed",
            "glowing",
            "glows",
            "flash",
            "flashed",
            "flashing",
            "dim",
            "dimmed",
            "dimming",
            "dimly",
            "gleam",
            "gleamed",
            "gleaming",
            "sparkle",
            "sparkled",
            "sparkling",
            "vivid",
            "vividly",
            "pale",
            "paler",
            "palest",
            "colorful",
            "colourful",
            "faded",
            "fading",
            "fade",
            "blinding",
            "blinded",
            "blind",
            "translucent",
            "opaque",
        ];
        let sound: &[&str] = &[
            "hear",
            "heard",
            "hearing",
            "hears",
            "listen",
            "listened",
            "listening",
            "listens",
            "ring",
            "rang",
            "rung",
            "ringing",
            "rings",
            "echo",
            "echoed",
            "echoing",
            "echoes",
            "whisper",
            "whispered",
            "whispering",
            "whispers",
            "shout",
            "shouted",
            "shouting",
            "shouts",
            "crash",
            "crashed",
            "crashing",
            "crashes",
            "bang",
            "banged",
            "banging",
            "bangs",
            "murmur",
            "murmured",
            "murmuring",
            "murmurs",
            "hum",
            "hummed",
            "humming",
            "hums",
            "buzz",
            "buzzed",
            "buzzing",
            "roar",
            "roared",
            "roaring",
            "roars",
            "crack",
            "cracked",
            "cracking",
            "cracks",
            "rustle",
            "rustled",
            "rustling",
            "thunder",
            "thundered",
            "thundering",
            "silence",
            "silenced",
            "silencing",
            "silent",
            "silently",
            "quiet",
            "quieter",
            "quietest",
            "quietly",
            "loud",
            "louder",
            "loudest",
            "loudly",
            "screech",
            "screeched",
            "screeching",
            "clatter",
            "clattered",
            "clattering",
            "chime",
            "chimed",
            "chiming",
            "chimes",
            "rumble",
            "rumbled",
            "rumbling",
        ];
        let smell: &[&str] = &[
            "smell",
            "smelled",
            "smelling",
            "smells",
            "smelt",
            "scent",
            "scented",
            "scents",
            "odor",
            "odors",
            "odour",
            "odours",
            "aroma",
            "aromas",
            "aromatic",
            "stench",
            "stenches",
            "fragrance",
            "fragrances",
            "fragrant",
            "whiff",
            "whiffed",
            "reek",
            "reeked",
            "reeking",
            "reeks",
            "perfume",
            "perfumed",
            "perfumes",
            "musty",
            "acrid",
            "pungent",
        ];
        let touch: &[&str] = &[
            "touch",
            "touched",
            "touching",
            "touches",
            "feel",
            "feeling",
            "feels",
            "felt",
            "grip",
            "gripped",
            "gripping",
            "grips",
            "grab",
            "grabbed",
            "grabbing",
            "grabs",
            "stroke",
            "stroked",
            "stroking",
            "strokes",
            "smooth",
            "smoother",
            "smoothest",
            "smoothly",
            "rough",
            "rougher",
            "roughest",
            "roughly",
            "cold",
            "colder",
            "coldest",
            "coldly",
            "coldness",
            "warm",
            "warmer",
            "warmest",
            "warmly",
            "warmth",
            "hot",
            "hotter",
            "hottest",
            "soft",
            "softer",
            "softest",
            "softly",
            "softness",
            "hard",
            "harder",
            "hardest",
            "sharp",
            "sharper",
            "sharpest",
            "sharply",
            "wet",
            "wetter",
            "wettest",
            "dry",
            "drier",
            "driest",
            "sticky",
            "prickle",
            "prickled",
            "prickling",
            "prickly",
            "sting",
            "stung",
            "stinging",
            "stings",
            "caress",
            "caressed",
            "caressing",
            "brush",
            "brushed",
            "brushing",
            "texture",
            "textured",
            "textures",
        ];
        let taste: &[&str] = &[
            "taste",
            "tasted",
            "tasting",
            "tastes",
            "tasteless",
            "flavor",
            "flavored",
            "flavors",
            "flavour",
            "flavoured",
            "flavours",
            "sweet",
            "sweeter",
            "sweetest",
            "sweetly",
            "sweetness",
            "bitter",
            "bitterly",
            "bitterness",
            "sour",
            "soured",
            "sourly",
            "salty",
            "saltier",
            "saltiest",
            "savory",
            "savoury",
            "spicy",
            "spicier",
            "spiciest",
            "bland",
            "blander",
            "blandest",
            "delicious",
            "deliciously",
            "mouth",
            "mouths",
            "mouthful",
            "tongue",
            "tongues",
            "chew",
            "chewed",
            "chewing",
            "chews",
            "swallow",
            "swallowed",
            "swallowing",
            "swallows",
            "gulp",
            "gulped",
            "gulping",
            "gulps",
            "sip",
            "sipped",
            "sipping",
            "sips",
        ];

        let mut patterns: Vec<&str> = Vec::new();
        let mut senses: Vec<&str> = Vec::new();

        for (words, sense) in [
            (sight as &[&str], "sight"),
            (sound, "sound"),
            (smell, "smell"),
            (touch, "touch"),
            (taste, "taste"),
        ] {
            for w in words {
                patterns.push(w);
                senses.push(sense);
            }
        }

        let automaton = AhoCorasick::builder()
            .match_kind(aho_corasick::MatchKind::LeftmostLongest)
            .build(&patterns)
            .expect("failed to build AhoCorasick");
        SensoryMatcher {
            automaton,
            sense_map: senses,
        }
    })
}

// ---------------------------------------------------------------------------
// count_syllables
// ---------------------------------------------------------------------------

/// High-accuracy syllable counter. Uses a static override map for common
/// English words and handles silent-e patterns and multi-vowel exceptions.
pub fn count_syllables(word: &str) -> usize {
    let lower = word.trim().to_lowercase();
    if lower.is_empty() {
        return 0;
    }

    let alpha: String = lower.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    if alpha.is_empty() {
        return 0;
    }

    if let Some(&count) = common_overrides().get(alpha.as_str()) {
        return count;
    }

    let chars: Vec<char> = alpha.chars().collect();
    let len = chars.len();
    let mut adjustment: i32 = 0;

    // Multi-letter sequence adjustments.
    // These correct for vowel pairs that the counter treats as one group
    // but are actually pronounced as separate syllables, or vice versa.
    // "ia" in words like "official" = 2 syllables, but counted as 1 group -> add 1
    if alpha.contains("ia") && !alpha.contains("tia") && !alpha.contains("ial") {
        adjustment += 1;
    }
    // "eous" = 2 syllables (e-ous), often counted as 1 -> add 1
    if alpha.contains("eous") {
        adjustment += 1;
    }
    // "ious" = 2 syllables (i-ous), often counted as 1 -> add 1
    if alpha.contains("ious") && !alpha.contains("tious") {
        adjustment += 1;
    }

    // Prepare working copy: strip trailing "e" if len > 2 and not ending "le"
    let mut working = chars.clone();
    if len > 2 && working.last() == Some(&'e') && !alpha.ends_with("le") {
        working.pop();
    }

    // Count vowel groups
    let mut count: i32 = 0;
    let mut in_vowel = false;
    for (i, &ch) in working.iter().enumerate() {
        let is_vowel = match ch {
            'a' | 'e' | 'i' | 'o' | 'u' => true,
            'y' if i > 0 => true,
            _ => false,
        };
        if is_vowel {
            if !in_vowel {
                count += 1;
                in_vowel = true;
            }
        } else {
            in_vowel = false;
        }
    }

    // Silent-e suffix patterns
    for suffix in &["ake", "ine", "ore", "ure", "ise"] {
        if alpha.ends_with(suffix) && len > 3 {
            adjustment -= 1;
            break;
        }
    }

    count += adjustment;
    std::cmp::max(count, 1) as usize
}

// ---------------------------------------------------------------------------
// count_sentences
// ---------------------------------------------------------------------------

/// Accurate sentence counter. Handles abbreviations, decimal numbers,
/// ellipses, Unicode ellipsis, and combined punctuation like "?!".
pub fn count_sentences(text: &str) -> usize {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return 0;
    }

    let re = sentence_regex();
    let mut count = 0;
    for caps in re.captures_iter(trimmed) {
        if caps.get(1).is_some() {
            count += 1;
        }
    }

    std::cmp::max(count, 1)
}

// ---------------------------------------------------------------------------
// estimate_dialogue_tokens
// ---------------------------------------------------------------------------

/// Count words inside quoted text. Supports straight double quotes, curly
/// double quotes, curly single quotes, and multi-paragraph dialogue.
pub fn estimate_dialogue_tokens(text: &str) -> usize {
    let re = dialogue_regex();
    let ws_re = word_split_regex();
    let mut total = 0;

    for caps in re.captures_iter(text) {
        for i in 1..=4 {
            if let Some(m) = caps.get(i) {
                total += ws_re.split(m.as_str()).filter(|s| !s.is_empty()).count();
            }
        }
    }

    total
}

// ---------------------------------------------------------------------------
// tokenize_words
// ---------------------------------------------------------------------------

/// Fast word tokenizer. Splits on whitespace/punctuation, preserves internal
/// apostrophes, lowercases all tokens, filters empties.
pub fn tokenize_words(text: &str) -> Vec<String> {
    let re = word_split_regex();
    re.split(text)
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_matches('\'').to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
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
// detect_sensory_words
// ---------------------------------------------------------------------------

/// Scan pre-tokenized words and return counts per sensory category (sight,
/// sound, smell, touch, taste). Uses AhoCorasick for O(n) matching with
/// whole-word enforcement.
pub fn detect_sensory_words(tokens: Vec<String>) -> HashMap<String, usize> {
    let matcher = sensory_matcher();
    let mut counts: HashMap<String, usize> = HashMap::new();
    counts.insert("sight".to_string(), 0);
    counts.insert("sound".to_string(), 0);
    counts.insert("smell".to_string(), 0);
    counts.insert("touch".to_string(), 0);
    counts.insert("taste".to_string(), 0);

    for token in &tokens {
        for mat in matcher.automaton.find_iter(token.as_str()) {
            let pattern_idx = mat.pattern().as_usize();
            let pattern_len = mat.end() - mat.start();
            if pattern_len == token.len() {
                let sense = matcher.sense_map[pattern_idx];
                if let Some(count) = counts.get_mut(sense) {
                    *count += 1;
                }
                break;
            }
        }
    }

    counts
}

// ---------------------------------------------------------------------------
// analyze_atmosphere
// ---------------------------------------------------------------------------

use std::collections::HashSet;

fn grounding_words() -> &'static HashSet<&'static str> {
    static INST: OnceLock<HashSet<&str>> = OnceLock::new();
    INST.get_or_init(|| {
        HashSet::from([
            "room",
            "house",
            "door",
            "window",
            "wall",
            "floor",
            "ceiling",
            "table",
            "chair",
            "bed",
            "street",
            "road",
            "path",
            "garden",
            "forest",
            "river",
            "lake",
            "mountain",
            "hill",
            "valley",
            "field",
            "ocean",
            "sea",
            "beach",
            "cliff",
            "cave",
            "bridge",
            "castle",
            "tower",
            "church",
            "city",
            "town",
            "village",
            "rain",
            "snow",
            "wind",
            "sun",
            "moon",
            "star",
            "sky",
            "cloud",
            "fog",
            "mist",
            "storm",
            "lightning",
            "dusk",
            "dawn",
            "sunset",
            "sunrise",
            "tree",
            "grass",
            "stone",
            "rock",
            "sand",
            "mud",
            "dust",
            "lamp",
            "candle",
            "fire",
            "smoke",
            "curtain",
            "carpet",
            "stair",
            // Body parts
            "hand",
            "hands",
            "face",
            "faces",
            "eye",
            "eyes",
            "body",
            "arm",
            "arms",
            "finger",
            "fingers",
            "head",
            "heart",
            "skin",
            "hair",
            "shoulder",
            "lip",
            "lips",
            // Clothing
            "clothes",
            "dress",
            "shirt",
            "coat",
            "cloak",
            "hat",
            "boots",
            "shoes",
            "robe",
            "gown",
            // Objects
            "glass",
            "cup",
            "plate",
            "food",
            "drink",
            "sword",
            "knife",
            "gun",
            "weapon",
            "book",
            "letter",
            "key",
            "ring",
            "torch",
            "lantern",
            "mirror",
            "clock",
            // Visual/atmospheric
            "light",
            "shadow",
            "darkness",
            "color",
            "air",
            "breath",
            "scent",
            "voice",
            "sound",
        ])
    })
}

#[derive(Clone, Debug)]
pub struct TemporalGrounding {
    pub time_of_day: HashMap<String, usize>,
    pub weather: HashMap<String, usize>,
}

#[derive(Clone, Debug)]
pub struct EmotionalAtmosphere {
    pub positive: usize,
    pub negative: usize,
    pub neutral: usize,
    pub dominant_tone: String,
}

#[derive(Clone, Debug)]
pub struct AtmosphereResult {
    pub score: f64,
    pub is_white_room: bool,
    pub sensory_counts: HashMap<String, usize>,
    pub grounding_density: f64,
    pub dominant_sense: String,
    pub grounding_word_count: usize,
    pub total_tokens: usize,
    pub temporal_grounding: TemporalGrounding,
    pub emotional_atmosphere: EmotionalAtmosphere,
    pub sensory_balance_score: f64,
}

const WHITE_ROOM_THRESHOLD: f64 = 0.02;

const TIME_WORDS: &[(&str, &str)] = &[
    ("morning", "morning"),
    ("noon", "noon"),
    ("afternoon", "afternoon"),
    ("evening", "evening"),
    ("night", "night"),
    ("midnight", "midnight"),
    ("dawn", "dawn"),
    ("dusk", "dusk"),
];

const WEATHER_WORDS: &[(&str, &str)] = &[
    ("rain", "rain"),
    ("storm", "storm"),
    ("snow", "snow"),
    ("wind", "wind"),
    ("fog", "fog"),
    ("sun", "sun"),
    ("cloud", "cloud"),
    ("mist", "mist"),
    ("lightning", "lightning"),
    ("hail", "hail"),
    ("thunder", "thunder"),
];

const POSITIVE_ATMOSPHERE: &[&str] = &[
    "welcoming",
    "peaceful",
    "vibrant",
    "serene",
    "warm",
    "cheerful",
    "bright",
    "cozy",
    "inviting",
    "gentle",
    "lush",
    "radiant",
];

const NEGATIVE_ATMOSPHERE: &[&str] = &[
    "oppressive",
    "eerie",
    "desolate",
    "suffocating",
    "menacing",
    "bleak",
    "grim",
    "foreboding",
    "dreary",
    "barren",
    "haunting",
    "ominous",
];

const NEUTRAL_ATMOSPHERE: &[&str] = &[
    "chaotic", "still", "vast", "quiet", "sparse", "dense", "hushed", "remote",
];

fn compute_sensory_balance(sensory_counts: &HashMap<String, usize>) -> f64 {
    let total: usize = sensory_counts.values().sum();
    if total == 0 {
        return 0.0;
    }
    let n_senses = sensory_counts.len() as f64;
    let even_fraction = 1.0 / n_senses;
    let max_fraction = *sensory_counts.values().max().unwrap_or(&0) as f64 / total as f64;
    (1.0 - (max_fraction - even_fraction)).clamp(0.0, 1.0)
}

pub fn analyze_atmosphere(scene_text: &str) -> AtmosphereResult {
    let tokens = tokenize_words(scene_text);
    let total_tokens = tokens.len();
    let sensory_counts = detect_sensory_words(tokens.clone());

    let grounding_set = grounding_words();
    let grounding_word_count = tokens
        .iter()
        .filter(|t| grounding_set.contains(t.as_str()))
        .count();

    let sensory_total: usize = sensory_counts.values().sum();

    let grounding_density = if total_tokens > 0 {
        (sensory_total + grounding_word_count) as f64 / total_tokens as f64
    } else {
        0.0
    };

    let score = (grounding_density / WHITE_ROOM_THRESHOLD).min(1.0);
    let is_white_room = grounding_density < WHITE_ROOM_THRESHOLD;

    let dominant_sense = sensory_counts
        .iter()
        .max_by_key(|(_, v)| **v)
        .map(|(k, _)| k.clone())
        .unwrap_or_else(|| "sight".to_string());

    // Temporal grounding
    let mut time_of_day: HashMap<String, usize> = HashMap::new();
    let mut weather: HashMap<String, usize> = HashMap::new();
    for t in &tokens {
        for &(word, key) in TIME_WORDS {
            if t == word {
                *time_of_day.entry(key.to_string()).or_insert(0) += 1;
            }
        }
        for &(word, key) in WEATHER_WORDS {
            if t == word {
                *weather.entry(key.to_string()).or_insert(0) += 1;
            }
        }
    }

    // Emotional atmosphere
    let mut positive = 0usize;
    let mut negative = 0usize;
    let mut neutral = 0usize;
    for t in &tokens {
        let s = t.as_str();
        if POSITIVE_ATMOSPHERE.contains(&s) {
            positive += 1;
        } else if NEGATIVE_ATMOSPHERE.contains(&s) {
            negative += 1;
        } else if NEUTRAL_ATMOSPHERE.contains(&s) {
            neutral += 1;
        }
    }
    let dominant_tone = if positive >= negative && positive >= neutral {
        "positive"
    } else if negative >= positive && negative >= neutral {
        "negative"
    } else {
        "neutral"
    }
    .to_string();

    let sensory_balance_score = compute_sensory_balance(&sensory_counts);

    AtmosphereResult {
        score,
        is_white_room,
        sensory_counts,
        grounding_density,
        dominant_sense,
        grounding_word_count,
        total_tokens,
        temporal_grounding: TemporalGrounding {
            time_of_day,
            weather,
        },
        emotional_atmosphere: EmotionalAtmosphere {
            positive,
            negative,
            neutral,
            dominant_tone,
        },
        sensory_balance_score,
    }
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

/// Split `text` into paragraphs, the canonical (§6.1) definition shared across
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

/// Count the paragraphs in `text` per the canonical [`split_paragraphs`] rule.
pub fn count_paragraphs(text: &str) -> usize {
    split_paragraphs(text).len()
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
    // 1. Byte spans of the trimmed, non-empty blocks (same boundaries the
    //    string-only segmenter produces).
    let mut byte_spans: Vec<(usize, usize)> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    let mut byte = 0usize;
    for piece in text.split("\n\n") {
        let piece_start = byte;
        // Advance past this piece and its "\n\n" separator; the overshoot on the
        // final piece is never read.
        byte += piece.len() + 2;
        let trimmed = piece.trim();
        if trimmed.is_empty() {
            continue;
        }
        let lead = piece.len() - piece.trim_start().len();
        let start = piece_start + lead;
        byte_spans.push((start, start + trimmed.len()));
        texts.push(trimmed.to_string());
    }
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
                let snippet = if sent.len() > 120 {
                    format!("{}...", &sent[..floor_char_boundary(sent, 120)])
                } else {
                    sent.to_string()
                };
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
                let snippet = if sent.len() > 120 {
                    format!("{}...", &sent[..floor_char_boundary(sent, 120)])
                } else {
                    sent.to_string()
                };
                results.push(ProseIssue {
                    scene_id,
                    issue_type: format!("show_dont_tell_{}", pat.ptype),
                    snippet,
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

fn word_tokenize_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[a-z]+").expect("word tokenize regex"))
}

pub fn detect_word_echoes(scene_text: &str, scene_id: usize) -> Vec<ProseIssue> {
    const WINDOW_SIZE: usize = 100;
    const STEP: usize = 50;
    const MIN_WORD_LEN: usize = 5;
    const MIN_REPEATS: usize = 3;
    const MAX_ISSUES: usize = 10;

    let lower = scene_text.to_lowercase();
    let re = word_tokenize_re();
    let words: Vec<&str> = re.find_iter(&lower).map(|m| m.as_str()).collect();

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
        for (&w, &count) in &freq {
            if results.len() >= MAX_ISSUES {
                break;
            }
            if count >= MIN_REPEATS && !seen_words.contains(w) {
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
    if byte_offset > s.len() {
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
        assert_eq!(count_paragraphs("one\r\n\r\ntwo\n \nthree"), 3);
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

    // -- detect_sensory_words --

    #[test]
    fn test_sensory_sight() {
        let tokens: Vec<String> = vec![
            "she".into(),
            "saw".into(),
            "a".into(),
            "bright".into(),
            "glow".into(),
        ];
        let result = detect_sensory_words(tokens);
        assert_eq!(result["sight"], 3);
        assert_eq!(result["sound"], 0);
    }

    #[test]
    fn test_sensory_sound() {
        let tokens: Vec<String> = vec![
            "the".into(),
            "thunder".into(),
            "crashed".into(),
            "loudly".into(),
        ];
        let result = detect_sensory_words(tokens);
        assert_eq!(result["sound"], 3);
    }

    #[test]
    fn test_sensory_mixed() {
        let tokens: Vec<String> = vec![
            "she".into(),
            "heard".into(),
            "the".into(),
            "sweet".into(),
            "smell".into(),
        ];
        let result = detect_sensory_words(tokens);
        assert_eq!(result["sound"], 1);
        assert_eq!(result["taste"], 1);
        assert_eq!(result["smell"], 1);
    }

    #[test]
    fn test_sensory_empty() {
        let result = detect_sensory_words(vec![]);
        assert_eq!(result["sight"], 0);
        assert_eq!(result["sound"], 0);
        assert_eq!(result["smell"], 0);
        assert_eq!(result["touch"], 0);
        assert_eq!(result["taste"], 0);
    }

    #[test]
    fn test_sensory_no_partial_match() {
        let tokens = vec!["hardly".to_string()];
        let result = detect_sensory_words(tokens);
        assert_eq!(result["touch"], 0);
    }

    #[test]
    fn test_sensory_all_senses() {
        let tokens: Vec<String> = vec![
            "bright".into(),
            "whisper".into(),
            "fragrant".into(),
            "smooth".into(),
            "sweet".into(),
        ];
        let result = detect_sensory_words(tokens);
        assert_eq!(result["sight"], 1);
        assert_eq!(result["sound"], 1);
        assert_eq!(result["smell"], 1);
        assert_eq!(result["touch"], 1);
        assert_eq!(result["taste"], 1);
    }

    // -- analyze_atmosphere --

    #[test]
    fn test_atmosphere_rich_scene() {
        let text = "The bright sun cast shadows across the stone wall. \
                    Wind rustled through the tall grass, carrying the sweet fragrance \
                    of wildflowers. She felt the warm rough texture of the rock beneath \
                    her fingers as thunder rumbled in the distance.";
        let result = analyze_atmosphere(text);
        assert!(
            result.score > 0.5,
            "rich scene should score above 0.5, got {}",
            result.score
        );
        assert!(!result.is_white_room);
        assert!(result.grounding_word_count > 0);
        assert!(result.total_tokens > 0);
        assert!(result.grounding_density > 0.0);
    }

    #[test]
    fn test_atmosphere_white_room() {
        let text = "He thought about the situation. She considered the implications. \
                    They discussed the problem and came to a conclusion. \
                    The idea was interesting but the logic was flawed.";
        let result = analyze_atmosphere(text);
        assert!(result.is_white_room, "abstract text should be white room");
        assert!(result.score < 1.0);
    }

    #[test]
    fn test_atmosphere_empty() {
        let result = analyze_atmosphere("");
        assert_eq!(result.total_tokens, 0);
        assert_eq!(result.score, 0.0);
        assert!(result.is_white_room);
        assert_eq!(result.grounding_density, 0.0);
    }

    #[test]
    fn test_atmosphere_dominant_sense() {
        let text = "She saw the bright glow shimmering in the dark room.";
        let result = analyze_atmosphere(text);
        assert_eq!(result.dominant_sense, "sight");
    }

    #[test]
    fn test_atmosphere_grounding_words_counted() {
        let text = "The castle door opened to reveal a stone stair leading to the tower.";
        let result = analyze_atmosphere(text);
        assert!(
            result.grounding_word_count >= 4,
            "expected at least 4 grounding words, got {}",
            result.grounding_word_count
        );
    }

    #[test]
    fn test_temporal_grounding_time() {
        let text = "The morning sun rose. By evening the sky had darkened.";
        let result = analyze_atmosphere(text);
        assert_eq!(
            result.temporal_grounding.time_of_day.get("morning"),
            Some(&1)
        );
        assert_eq!(
            result.temporal_grounding.time_of_day.get("evening"),
            Some(&1)
        );
    }

    #[test]
    fn test_temporal_grounding_weather() {
        let text = "Rain pounded the roof. The storm grew stronger with wind and lightning.";
        let result = analyze_atmosphere(text);
        assert_eq!(result.temporal_grounding.weather.get("rain"), Some(&1));
        assert_eq!(result.temporal_grounding.weather.get("storm"), Some(&1));
        assert_eq!(result.temporal_grounding.weather.get("wind"), Some(&1));
        assert_eq!(result.temporal_grounding.weather.get("lightning"), Some(&1));
    }

    #[test]
    fn test_emotional_atmosphere_negative() {
        let text = "The oppressive darkness felt eerie and menacing in the desolate hallway.";
        let result = analyze_atmosphere(text);
        assert!(result.emotional_atmosphere.negative > 0);
        assert_eq!(result.emotional_atmosphere.dominant_tone, "negative");
    }

    #[test]
    fn test_emotional_atmosphere_positive() {
        let text = "The peaceful garden was serene and welcoming with vibrant flowers.";
        let result = analyze_atmosphere(text);
        assert!(result.emotional_atmosphere.positive > 0);
        assert_eq!(result.emotional_atmosphere.dominant_tone, "positive");
    }

    #[test]
    fn test_sensory_balance_all_sight() {
        let tokens: Vec<String> = vec![
            "bright".into(),
            "glow".into(),
            "shimmer".into(),
            "dark".into(),
            "shadow".into(),
        ];
        let counts = detect_sensory_words(tokens);
        let balance = compute_sensory_balance(&counts);
        // All sight = unbalanced, should be low
        assert!(
            balance < 0.5,
            "all-sight balance should be low, got {}",
            balance
        );
    }

    #[test]
    fn test_sensory_balance_even() {
        let tokens: Vec<String> = vec![
            "bright".into(),
            "whisper".into(),
            "fragrant".into(),
            "smooth".into(),
            "sweet".into(),
        ];
        let counts = detect_sensory_words(tokens);
        let balance = compute_sensory_balance(&counts);
        // One per sense = very balanced
        assert!(
            balance > 0.7,
            "even distribution should score high, got {}",
            balance
        );
    }

    #[test]
    fn test_sensory_balance_empty() {
        let counts = detect_sensory_words(vec![]);
        let balance = compute_sensory_balance(&counts);
        assert_eq!(balance, 0.0);
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
        assert!(issues
            .iter()
            .any(|i| i.issue_type == "show_dont_tell_narrative_distance"));
    }

    #[test]
    fn test_show_dont_tell_advanced_editorial() {
        let issues = detect_show_dont_tell("It was clear that he would not return.", 0);
        assert!(issues
            .iter()
            .any(|i| i.issue_type == "show_dont_tell_editorial"));
    }

    #[test]
    fn test_show_dont_tell_advanced_filter_word() {
        let issues = detect_show_dont_tell("She could see the flames rising.", 0);
        assert!(issues
            .iter()
            .any(|i| i.issue_type == "show_dont_tell_filter_word"));
    }

    #[test]
    fn test_show_dont_tell_advanced_diluted() {
        let issues = detect_show_dont_tell("He started to run toward the exit.", 0);
        assert!(issues
            .iter()
            .any(|i| i.issue_type == "show_dont_tell_diluted_action"));
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
}
