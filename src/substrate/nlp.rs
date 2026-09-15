use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// Calculates TF-IDF for a collection of documents (scenes).
pub fn compute_tfidf(
    documents: Vec<Vec<String>>,
) -> Result<Vec<HashMap<String, f64>>, crate::error::Error> {
    let num_docs = documents.len();
    if num_docs == 0 {
        return Ok(vec![]);
    }

    let mut df: HashMap<String, usize> = HashMap::new();
    let mut doc_tfs: Vec<HashMap<String, usize>> = Vec::with_capacity(num_docs);

    for doc in &documents {
        let mut tf: HashMap<String, usize> = HashMap::new();
        let mut unique_terms: std::collections::HashSet<&String> = std::collections::HashSet::new();
        for term in doc {
            *tf.entry(term.clone()).or_insert(0) += 1;
            unique_terms.insert(term);
        }
        for term in unique_terms {
            *df.entry(term.clone()).or_insert(0) += 1;
        }
        doc_tfs.push(tf);
    }

    let mut results = Vec::with_capacity(num_docs);
    let num_docs_f = num_docs as f64;

    for tf_map in doc_tfs {
        let mut tfidf_map = HashMap::new();
        let total_terms: usize = tf_map.values().sum();
        let total_terms_f = total_terms as f64;

        for (term, count) in tf_map {
            let tf = (count as f64) / total_terms_f;
            let doc_freq = *df.get(&term).unwrap_or(&1) as f64;
            let idf = (num_docs_f / doc_freq).ln();
            tfidf_map.insert(term, tf * idf);
        }
        results.push(tfidf_map);
    }

    Ok(results)
}

/// Calculates agency ratio for a protagonist based on dependency relations.
pub fn calculate_agency_ratio(
    relations: Vec<(String, String, String)>,
    protagonist_name: String,
) -> Result<f64, crate::error::Error> {
    let mut agent_count = 0;
    let mut patient_count = 0;
    let name_lower = protagonist_name.to_lowercase();

    for (_head, dep, child) in relations {
        let child_lower = child.to_lowercase();
        let matches_name = child_lower
            .split_whitespace()
            .any(|word| word == name_lower);
        if matches_name {
            match dep.as_str() {
                "nsubj" => agent_count += 1,
                "dobj" | "pobj" | "iobj" | "nsubjpass" => patient_count += 1,
                _ => {}
            }
        }
    }

    let total = agent_count + patient_count;
    if total == 0 {
        return Err(crate::error::Error::AnalysisFailed(
            "no agency relations found for protagonist".into(),
        ));
    }
    Ok((agent_count as f64) / (total as f64))
}

/// Calculates micro-tension based on syntactic structure.
pub fn calculate_micro_tension(root_indices: Vec<usize>) -> Result<f64, crate::error::Error> {
    if root_indices.is_empty() {
        return Ok(0.0);
    }
    let sum: usize = root_indices.iter().sum();
    Ok((sum as f64) / (root_indices.len() as f64))
}

/// Calculates a 102-dimension stylometric vector.
/// [0-99]: Relative frequencies of 100 function words
/// [100]: Average sentence length (normalized)
/// [101]: Type-token ratio
pub fn calculate_stylometric_vector(
    tokens: Vec<String>,
    function_words: Vec<String>,
    avg_sentence_length: f64,
) -> Result<Vec<f64>, crate::error::Error> {
    let mut vector = vec![0.0; 102];
    let total_tokens = tokens.len() as f64;

    if total_tokens == 0.0 {
        return Ok(vector);
    }

    // 1. Function word frequencies
    let mut counts: HashMap<String, usize> = HashMap::new();
    for token in &tokens {
        *counts.entry(token.to_lowercase()).or_insert(0) += 1;
    }

    for (i, word) in function_words.iter().enumerate() {
        if i >= 100 {
            break;
        }
        let count = *counts.get(&word.to_lowercase()).unwrap_or(&0) as f64;
        vector[i] = count / total_tokens;
    }

    // 2. Average sentence length (normalized by 50)
    vector[100] = avg_sentence_length / 50.0;

    // 3. Type-Token Ratio
    let unique_types = counts.len() as f64;
    vector[101] = unique_types / total_tokens;

    Ok(vector)
}

// ---------------------------------------------------------------------------
// Static initializers for character extraction
// ---------------------------------------------------------------------------

fn false_positives() -> &'static HashSet<String> {
    static FP: OnceLock<HashSet<String>> = OnceLock::new();
    FP.get_or_init(|| {
        [
            "The",
            "This",
            "That",
            "Then",
            "There",
            "They",
            "When",
            "What",
            "Where",
            "With",
            "From",
            "Into",
            "Upon",
            "Before",
            "After",
            "Chapter",
            "Scene",
            "Act",
            "Part",
            "She",
            "Her",
            "His",
            "Him",
            "Its",
            "How",
            "Why",
            "Who",
            "Which",
            "Each",
            "Every",
            "Another",
            "Other",
            "Some",
            "Any",
            "All",
            "Both",
            "Few",
            "Many",
            "Much",
            "More",
            "Most",
            "Such",
            "Not",
            "But",
            "And",
            // Pronouns: never characters, but capitalize at sentence start and so
            // slip past the sentence-start filter when they appear mid-clause too.
            "He",
            "It",
            "We",
            "You",
            "Me",
            "Us",
            "Them",
            "Their",
            "Our",
            "Your",
            "Himself",
            "Herself",
            "Themselves",
            // Common sentence-initial / function & connective words seen as noise
            // ("As", "So", "Yet", ...) — capitalized openings mistaken for names.
            "As",
            "So",
            "Yet",
            "For",
            "Nor",
            "Or",
            "If",
            "Than",
            "While",
            "Because",
            "Although",
            "Though",
            "Since",
            "Until",
            "Unless",
            "Whether",
            "Whenever",
            "Wherever",
            "However",
            "Meanwhile",
            "Now",
            "Here",
            "Once",
            "Soon",
            "Later",
            "Again",
            "Still",
            "Even",
            "Perhaps",
            "Maybe",
            "Indeed",
            "Instead",
            "Finally",
            "Suddenly",
            "Slowly",
        ]
        .iter()
        .map(|w| w.to_lowercase())
        .collect()
    })
}

/// A surface that survives the keep-rule must still plausibly be a name: at least
/// two characters (a lone capital is never a character — "I", a stray initial) and
/// not an all-caps token (acronyms, signage, emphatic shouts like "CITY"). Applied
/// to each whole name; a two-word name passes if each word is itself plausible.
fn is_plausible_name(name: &str) -> bool {
    name.split_whitespace().all(|word| {
        let len = word.chars().count();
        len >= 2 && !word.chars().all(|c| c.is_uppercase() || !c.is_alphabetic())
    })
}

fn dialogue_attr_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?:said|asked|replied|whispered|shouted|called|cried|muttered|exclaimed|answered|declared|insisted|demanded|pleaded|murmured|stammered|snapped|groaned|sighed)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
        ).expect("invalid dialogue attr regex")
    })
}

fn dialogue_attr_reverse_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:said|asked|replied|whispered|shouted|called|cried|muttered|exclaimed|answered|declared|insisted|demanded|pleaded|murmured|stammered|snapped|groaned|sighed)"
        ).expect("invalid dialogue attr reverse regex")
    })
}

fn capitalized_word_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[A-Z][a-z]+").expect("invalid capitalized word regex"))
}

fn capitalized_pair_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"[A-Z][a-z]+\s+[A-Z][a-z]+").expect("invalid capitalized pair regex")
    })
}

// ---------------------------------------------------------------------------
// edit_distance helper
// ---------------------------------------------------------------------------

fn edit_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[m][n]
}

/// Surfaces that appear in a dialogue-attribution frame ("X said" / "said X")
/// in any scene. Dialogue attribution is a high-precision *person* signal — a
/// place or organization never speaks — so callers use it to protect a name
/// from place/common-word vetoes even when the NER tagger misses the person.
/// Reuses the same two attribution regexes as [`extract_character_names_regex`].
pub fn dialogue_attributed_names<S: AsRef<str>>(scenes: &[S]) -> HashSet<String> {
    let fp = false_positives();
    let mut names = HashSet::new();
    for scene in scenes {
        let scene = scene.as_ref();
        for re in [dialogue_attr_regex(), dialogue_attr_reverse_regex()] {
            for cap in re.captures_iter(scene) {
                if let Some(m) = cap.get(1) {
                    let name = m.as_str();
                    if !fp.contains(&name.to_lowercase()) {
                        names.insert(name.to_string());
                    }
                }
            }
        }
    }
    names
}

// ---------------------------------------------------------------------------
// extract_character_names_regex
// ---------------------------------------------------------------------------

/// Tally each capture-group-1 name from `re` over `scene`, skipping known
/// false positives; every match also marks the name as a strong (high-precision)
/// signal. Shared by the forward and reverse dialogue-attribution passes.
fn count_attributions(
    re: &regex::Regex,
    scene: &str,
    fp: &HashSet<String>,
    counts: &mut HashMap<String, usize>,
    strong: &mut HashSet<String>,
) {
    for cap in re.captures_iter(scene) {
        if let Some(m) = cap.get(1) {
            let name = m.as_str().to_string();
            if !fp.contains(&name.to_lowercase()) {
                *counts.entry(name.clone()).or_insert(0) += 1;
                strong.insert(name);
            }
        }
    }
}

/// Regex-based character name extraction for when the NER model is unavailable.
/// Returns (name, count) pairs sorted by count descending.
pub fn extract_character_names_regex<S: AsRef<str>>(
    scenes: &[S],
) -> Result<Vec<(String, usize)>, crate::error::Error> {
    let fp = false_positives();
    let attr_re = dialogue_attr_regex();
    let attr_rev_re = dialogue_attr_reverse_regex();
    let cap_re = capitalized_word_regex();

    let mut counts: HashMap<String, usize> = HashMap::new();
    // Names with at least one dialogue attribution ("X said" / "said X"): a
    // high-precision signal, so these are kept even with few total mentions.
    let mut strong: HashSet<String> = HashSet::new();

    for scene in scenes {
        let scene = scene.as_ref();
        // Dialogue attribution ("X said" and "said X"): a strong person signal.
        count_attributions(attr_re, scene, fp, &mut counts, &mut strong);
        count_attributions(attr_rev_re, scene, fp, &mut counts, &mut strong);

        // Capitalized words NOT at sentence starts. Sentence-start offsets come
        // from the shared abbreviation-aware segmenter, not a fragile regex.
        let sentence_starts: HashSet<usize> = crate::substrate::utils::sentence_spans(scene)
            .into_iter()
            .map(|(s, _)| s)
            .collect();

        // Single capitalized words not at sentence starts
        for m in cap_re.find_iter(scene) {
            let start = m.start();
            if sentence_starts.contains(&start) {
                continue;
            }
            let name = m.as_str().to_string();
            if !fp.contains(&name.to_lowercase()) {
                *counts.entry(name).or_insert(0) += 1;
            }
        }

        // Two-word capitalized names (e.g. "Mary Smith")
        let pair_re = capitalized_pair_regex();
        for m in pair_re.find_iter(scene) {
            let start = m.start();
            if sentence_starts.contains(&start) {
                continue;
            }
            let name = m.as_str().to_string();
            let words: Vec<&str> = name.split_whitespace().collect();
            let all_fp = words.iter().all(|w| fp.contains(&w.to_lowercase()));
            if !all_fp {
                *counts.entry(name).or_insert(0) += 1;
            }
        }
    }

    // Keep a name with a strong (dialogue-attribution) signal regardless of
    // count, or any name reaching the weak-signal threshold of 3; sort by count.
    // A surface must also look like a name (`is_plausible_name`) — single-letter
    // and all-caps tokens are never characters and are dropped even if frequent.
    let mut result: Vec<(String, usize)> = counts
        .into_iter()
        .filter(|(name, c)| is_plausible_name(name) && (strong.contains(name) || *c >= 3))
        .collect();
    result.sort_by_key(|x| std::cmp::Reverse(x.1));

    Ok(result)
}

// ---------------------------------------------------------------------------
// extract_character_mentions
// ---------------------------------------------------------------------------

/// A single located name occurrence: its surface form, which scene it appears
/// in, and its byte offset within that scene's string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NameMention {
    pub surface: String,
    pub scene_index: usize,
    pub byte_offset: usize,
}

/// Per-occurrence companion to [`extract_character_names_regex`]: every located
/// mention, with offsets, whose surface survives the *same* keep-rule that
/// function applies. Entity resolution's context-embedding path needs the
/// position of each mention to embed its surrounding text; the aggregated
/// `(name, count)` form discards positions.
///
/// The set of surfaces returned here is identical to
/// [`extract_character_names_regex`] — it is reused verbatim to derive the
/// survivor set, so which names resolve never diverges between the two. The
/// difference is one entry per occurrence (in scene-then-offset order) instead
/// of a single aggregated count. A name kept only by a dialogue-attribution
/// signal still yields its attribution occurrences; offsets are byte offsets,
/// matching [`crate::substrate::utils::sentence_spans`].
pub fn extract_character_mentions<S: AsRef<str>>(
    scenes: &[S],
) -> Result<Vec<NameMention>, crate::error::Error> {
    // Reuse the canonical keep-rule verbatim: only surfaces it returns are kept.
    let survivors: HashSet<String> = extract_character_names_regex(scenes)?
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    if survivors.is_empty() {
        return Ok(Vec::new());
    }

    let fp = false_positives();
    let attr_re = dialogue_attr_regex();
    let attr_rev_re = dialogue_attr_reverse_regex();
    let cap_re = capitalized_word_regex();
    let pair_re = capitalized_pair_regex();

    let mut mentions: Vec<NameMention> = Vec::new();
    for (scene_index, scene) in scenes.iter().enumerate() {
        let scene = scene.as_ref();
        let sentence_starts: HashSet<usize> = crate::substrate::utils::sentence_spans(scene)
            .into_iter()
            .map(|(s, _)| s)
            .collect();

        let keep = |surface: &str, offset: usize, out: &mut Vec<NameMention>| {
            if survivors.contains(surface) {
                out.push(NameMention {
                    surface: surface.to_string(),
                    scene_index,
                    byte_offset: offset,
                });
            }
        };

        // Dialogue attributions (capture group 1 is the name), both directions.
        for re in [attr_re, attr_rev_re] {
            for cap in re.captures_iter(scene) {
                if let Some(m) = cap.get(1) {
                    let name = m.as_str();
                    if !fp.contains(&name.to_lowercase()) {
                        keep(name, m.start(), &mut mentions);
                    }
                }
            }
        }

        // Single capitalized words, not at sentence starts.
        for m in cap_re.find_iter(scene) {
            if sentence_starts.contains(&m.start()) {
                continue;
            }
            let name = m.as_str();
            if !fp.contains(&name.to_lowercase()) {
                keep(name, m.start(), &mut mentions);
            }
        }

        // Two-word capitalized names, not at sentence starts.
        for m in pair_re.find_iter(scene) {
            if sentence_starts.contains(&m.start()) {
                continue;
            }
            let name = m.as_str();
            let words: Vec<&str> = name.split_whitespace().collect();
            let all_fp = words.iter().all(|w| fp.contains(&w.to_lowercase()));
            if !all_fp {
                keep(name, m.start(), &mut mentions);
            }
        }
    }

    mentions.sort_by(|a, b| {
        a.scene_index
            .cmp(&b.scene_index)
            .then(a.byte_offset.cmp(&b.byte_offset))
    });
    Ok(mentions)
}

// ---------------------------------------------------------------------------
// deduplicate_characters
// ---------------------------------------------------------------------------

/// Merge near-duplicate character names using Levenshtein distance.
pub fn deduplicate_characters(names: Vec<String>) -> Result<Vec<String>, crate::error::Error> {
    // Sort by length descending (longer names take priority)
    let mut sorted = names;
    sorted.sort_by_key(|n| std::cmp::Reverse(n.len()));

    let mut accepted: Vec<String> = Vec::new();

    for name in &sorted {
        let name_lower = name.to_lowercase();
        let name_words: Vec<&str> = name.split_whitespace().collect();
        let mut is_dup = false;

        for existing in &accepted {
            let existing_lower = existing.to_lowercase();

            // Substring match: if one is contained in the other, merge (keep longer, which is already accepted)
            if existing_lower.contains(&name_lower) || name_lower.contains(&existing_lower) {
                is_dup = true;
                break;
            }

            // Shared first or last word
            let existing_words: Vec<&str> = existing.split_whitespace().collect();
            if let (Some(&n_first), Some(&n_last), Some(&e_first), Some(&e_last)) = (
                name_words.first(),
                name_words.last(),
                existing_words.first(),
                existing_words.last(),
            ) {
                let shares_first = n_first.eq_ignore_ascii_case(e_first);
                let shares_last = n_last.eq_ignore_ascii_case(e_last);
                if shares_first || shares_last {
                    is_dup = true;
                    break;
                }
            }

            // Normalized edit distance
            let max_len = name.len().max(existing.len());
            if max_len > 0 {
                let dist = edit_distance(&name_lower, &existing_lower);
                let normalized = dist as f64 / max_len as f64;
                if normalized < 0.25 {
                    is_dup = true;
                    break;
                }
            }
        }

        if !is_dup {
            accepted.push(name.clone());
        }
    }

    Ok(accepted)
}

// ---------------------------------------------------------------------------
// merge_character_aliases
// ---------------------------------------------------------------------------

/// Merge character aliases: if "Liz" and "Elizabeth" both appear,
/// merge under the most frequent form.
pub fn merge_character_aliases(
    counts: Vec<(String, usize)>,
) -> Result<Vec<(String, usize)>, crate::error::Error> {
    // Sort by count descending
    let mut sorted = counts;
    sorted.sort_by_key(|x| std::cmp::Reverse(x.1));

    // canonical_name -> total_count
    let mut canonical: Vec<(String, usize)> = Vec::new();

    for (name, count) in &sorted {
        let name_lower = name.to_lowercase();
        let name_words: Vec<&str> = name.split_whitespace().collect();
        let mut merged = false;

        for (canon, canon_count) in canonical.iter_mut() {
            let canon_lower = canon.to_lowercase();

            // Substring match
            if canon_lower.contains(&name_lower) || name_lower.contains(&canon_lower) {
                // Keep the most frequent variant; use length as tiebreaker
                if *count > *canon_count || (*count == *canon_count && name.len() > canon.len()) {
                    *canon = name.clone();
                }
                *canon_count += count;
                merged = true;
                break;
            }

            // Shared first or last word
            let canon_words: Vec<&str> = canon.split_whitespace().collect();
            if let (Some(&n_first), Some(&n_last), Some(&c_first), Some(&c_last)) = (
                name_words.first(),
                name_words.last(),
                canon_words.first(),
                canon_words.last(),
            ) {
                let shares_first = n_first.eq_ignore_ascii_case(c_first);
                let shares_last = n_last.eq_ignore_ascii_case(c_last);
                if shares_first || shares_last {
                    if *count > *canon_count || (*count == *canon_count && name.len() > canon.len())
                    {
                        *canon = name.clone();
                    }
                    *canon_count += count;
                    merged = true;
                    break;
                }
            }
        }

        if !merged {
            canonical.push((name.clone(), *count));
        }
    }

    // Sort final result by count descending
    canonical.sort_by_key(|x| std::cmp::Reverse(x.1));
    Ok(canonical)
}

// ---------------------------------------------------------------------------
// Theme keyword detection
// ---------------------------------------------------------------------------

type ThemeMap = Vec<(&'static str, Vec<&'static str>)>;

fn theme_keywords() -> &'static ThemeMap {
    static DATA: OnceLock<ThemeMap> = OnceLock::new();
    DATA.get_or_init(|| {
        vec![
            (
                "love",
                vec![
                    "love",
                    "loved",
                    "loving",
                    "heart",
                    "passion",
                    "desire",
                    "kiss",
                    "beloved",
                    "affection",
                    "romance",
                    "adore",
                ],
            ),
            (
                "death",
                vec![
                    "death", "die", "died", "dying", "kill", "killed", "dead", "grave", "funeral",
                    "corpse", "murder", "mortal",
                ],
            ),
            (
                "power",
                vec![
                    "power",
                    "control",
                    "authority",
                    "rule",
                    "king",
                    "queen",
                    "throne",
                    "command",
                    "dominion",
                    "tyrant",
                ],
            ),
            (
                "betrayal",
                vec![
                    "betray",
                    "betrayed",
                    "betrayal",
                    "traitor",
                    "deceive",
                    "deceived",
                    "treachery",
                    "deceit",
                ],
            ),
            (
                "redemption",
                vec![
                    "redeem",
                    "redeemed",
                    "redemption",
                    "forgive",
                    "forgiveness",
                    "atone",
                    "atonement",
                    "mercy",
                    "salvation",
                ],
            ),
            (
                "freedom",
                vec![
                    "freedom",
                    "escape",
                    "escaped",
                    "liberty",
                    "release",
                    "chain",
                    "chains",
                    "prison",
                    "captive",
                    "liberation",
                ],
            ),
            (
                "identity",
                vec![
                    "identity",
                    "mirror",
                    "mask",
                    "disguise",
                    "pretend",
                    "imposter",
                    "persona",
                    "transformation",
                ],
            ),
            (
                "justice",
                vec![
                    "justice",
                    "law",
                    "trial",
                    "judge",
                    "innocent",
                    "guilty",
                    "verdict",
                    "punishment",
                    "court",
                ],
            ),
            (
                "sacrifice",
                vec![
                    "sacrifice",
                    "sacrificed",
                    "surrender",
                    "surrendered",
                    "martyr",
                ],
            ),
            (
                "isolation",
                vec![
                    "alone",
                    "lonely",
                    "loneliness",
                    "solitude",
                    "exile",
                    "exiled",
                    "outcast",
                    "abandoned",
                    "forsaken",
                ],
            ),
            (
                "grief",
                vec![
                    "grief", "mourn", "mourning", "sorrow", "loss", "weep", "weeping", "lament",
                ],
            ),
            (
                "revenge",
                vec![
                    "revenge",
                    "vengeance",
                    "avenge",
                    "avenged",
                    "retribution",
                    "retaliate",
                    "grudge",
                ],
            ),
            (
                "hope",
                vec![
                    "hope", "hoped", "hopeful", "optimism", "dream", "dreamed", "aspire", "faith",
                    "believe",
                ],
            ),
            (
                "fear",
                vec![
                    "fear",
                    "feared",
                    "afraid",
                    "terror",
                    "dread",
                    "horror",
                    "panic",
                    "anxiety",
                    "nightmare",
                ],
            ),
            (
                "guilt",
                vec![
                    "guilt",
                    "guilty",
                    "shame",
                    "ashamed",
                    "remorse",
                    "regret",
                    "conscience",
                    "blame",
                ],
            ),
            (
                "coming_of_age",
                vec![
                    "grow",
                    "mature",
                    "childhood",
                    "innocence",
                    "adult",
                    "learn",
                    "mentor",
                    "first",
                    "realize",
                    "understand",
                    "independent",
                    "responsibility",
                ],
            ),
            (
                "war",
                vec![
                    "battle",
                    "soldier",
                    "army",
                    "weapon",
                    "enemy",
                    "fight",
                    "siege",
                    "victory",
                    "defeat",
                    "retreat",
                    "strategy",
                    "commander",
                ],
            ),
            (
                "corruption",
                vec![
                    "corrupt",
                    "bribe",
                    "scandal",
                    "dishonest",
                    "abuse",
                    "exploit",
                    "greed",
                    "scheme",
                    "conspiracy",
                ],
            ),
            (
                "survival",
                vec![
                    "survive",
                    "hunger",
                    "shelter",
                    "danger",
                    "wilderness",
                    "hunt",
                    "escape",
                    "endure",
                    "starve",
                    "struggle",
                ],
            ),
            (
                "family",
                vec![
                    "mother",
                    "father",
                    "sister",
                    "brother",
                    "daughter",
                    "son",
                    "family",
                    "home",
                    "inheritance",
                    "bloodline",
                    "legacy",
                ],
            ),
            (
                "class",
                vec![
                    "rich",
                    "poor",
                    "wealth",
                    "poverty",
                    "noble",
                    "servant",
                    "privilege",
                    "inequality",
                    "status",
                    "aristocrat",
                ],
            ),
            (
                "nature",
                vec![
                    "forest",
                    "river",
                    "mountain",
                    "ocean",
                    "wilderness",
                    "season",
                    "storm",
                    "garden",
                    "earth",
                    "sky",
                    "animal",
                ],
            ),
            (
                "technology",
                vec![
                    "machine",
                    "invention",
                    "progress",
                    "digital",
                    "artificial",
                    "robot",
                    "network",
                    "code",
                    "system",
                    "automation",
                ],
            ),
            (
                "faith",
                vec![
                    "god", "prayer", "church", "temple", "miracle", "divine", "sacred", "sin",
                    "heaven", "soul", "spirit", "blessing",
                ],
            ),
            (
                "madness",
                vec![
                    "mad",
                    "insane",
                    "hallucinate",
                    "delusion",
                    "paranoid",
                    "obsess",
                    "nightmare",
                    "voices",
                    "breakdown",
                    "asylum",
                ],
            ),
        ]
    })
}

/// Detect theme scores per scene based on keyword presence.
pub fn detect_themes_keywords(
    scenes_tokens: Vec<Vec<String>>,
) -> Result<Vec<HashMap<String, f64>>, crate::error::Error> {
    let themes = theme_keywords();
    let mut results = Vec::with_capacity(scenes_tokens.len());

    for tokens in &scenes_tokens {
        let word_count = tokens.len();
        let token_set: HashSet<&str> = tokens.iter().map(|t| t.as_str()).collect();
        let mut scene_themes = HashMap::new();

        for (theme_name, keywords) in themes {
            let hits = keywords.iter().filter(|kw| token_set.contains(*kw)).count();
            if hits > 0 {
                let coverage = hits as f64 / keywords.len() as f64;
                let density = if word_count > 0 {
                    (hits as f64 / (word_count as f64 / 100.0)).min(1.0)
                } else {
                    0.0
                };
                let score = (coverage + density) / 2.0;
                scene_themes.insert(theme_name.to_string(), score);
            }
        }

        results.push(scene_themes);
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Character co-occurrence edges
// ---------------------------------------------------------------------------

/// Build character co-occurrence edges from scene token lists.
/// Returns (char1, char2, weight) tuples sorted by weight descending.
pub fn build_cooccurrence_edges(
    scenes: Vec<Vec<String>>,
    characters: Vec<String>,
) -> Result<Vec<(String, String, usize)>, crate::error::Error> {
    // Build regex patterns for each character (case-insensitive word boundary)
    let patterns: Vec<(String, Regex)> = characters
        .iter()
        .map(|name| {
            (
                name.clone(),
                crate::substrate::text::word_boundary_regex(name),
            )
        })
        .collect();

    let mut edge_counts: HashMap<(String, String), usize> = HashMap::new();

    for scene_tokens in &scenes {
        let scene_text = scene_tokens.join(" ");
        let mut present: Vec<&str> = Vec::new();

        for (name, re) in &patterns {
            if re.is_match(&scene_text) {
                present.push(name);
            }
        }

        // Generate all unique pairs
        for i in 0..present.len() {
            for j in (i + 1)..present.len() {
                let (a, b) = if present[i] <= present[j] {
                    (present[i], present[j])
                } else {
                    (present[j], present[i])
                };
                *edge_counts
                    .entry((a.to_string(), b.to_string()))
                    .or_insert(0) += 1;
            }
        }
    }

    let mut edges: Vec<(String, String, usize)> = edge_counts
        .into_iter()
        .map(|((a, b), w)| (a, b, w))
        .collect();
    edges.sort_by_key(|x| std::cmp::Reverse(x.2));

    Ok(edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tfidf_basic() {
        let docs = vec![
            vec!["the".into(), "cat".into(), "sat".into()],
            vec!["the".into(), "dog".into(), "ran".into()],
        ];
        let result = compute_tfidf(docs).unwrap();
        assert_eq!(result.len(), 2);
        // "the" appears in both docs → IDF = ln(2/2) = 0, so TF-IDF = 0
        assert_eq!(result[0].get("the").copied().unwrap_or(1.0), 0.0);
        // "cat" appears in only doc 0 → IDF = ln(2/1) > 0
        assert!(result[0]["cat"] > 0.0);
        // "dog" appears in only doc 1
        assert!(result[1]["dog"] > 0.0);
    }

    #[test]
    fn test_tfidf_empty() {
        let result = compute_tfidf(vec![]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_agency_ratio_high() {
        let relations = vec![
            ("verb".into(), "nsubj".into(), "Alice".into()),
            ("verb".into(), "nsubj".into(), "Alice".into()),
            ("verb".into(), "dobj".into(), "Bob".into()),
        ];
        let ratio = calculate_agency_ratio(relations, "Alice".into()).unwrap();
        assert!(ratio > 0.5, "Expected > 0.5, got {ratio}");
    }

    #[test]
    fn test_agency_ratio_low() {
        let relations = vec![
            ("verb".into(), "dobj".into(), "Alice".into()),
            ("verb".into(), "dobj".into(), "Alice".into()),
            ("verb".into(), "nsubj".into(), "Alice".into()),
        ];
        let ratio = calculate_agency_ratio(relations, "Alice".into()).unwrap();
        assert!(ratio < 0.5, "Expected < 0.5, got {ratio}");
    }

    #[test]
    fn test_agency_ratio_missing_name() {
        let relations = vec![("verb".into(), "nsubj".into(), "Bob".into())];
        assert!(calculate_agency_ratio(relations, "Alice".into()).is_err());
    }

    #[test]
    fn test_agency_word_boundary() {
        // "John" should NOT match "Johnson"
        let relations = vec![("verb".into(), "nsubj".into(), "Johnson".into())];
        assert!(calculate_agency_ratio(relations, "John".into()).is_err());
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = crate::substrate::utils::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];
        let sim = crate::substrate::utils::cosine_similarity(&v1, &v2);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let v1 = vec![0.0, 0.0, 0.0];
        let v2 = vec![1.0, 2.0, 3.0];
        let sim = crate::substrate::utils::cosine_similarity(&v1, &v2);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_stylometric_vector_length() {
        let tokens: Vec<String> = vec!["the".into(), "a".into(), "cat".into()];
        let fw: Vec<String> = (0..100).map(|i| format!("fw{i}")).collect();
        let result = calculate_stylometric_vector(tokens, fw, 10.0).unwrap();
        assert_eq!(result.len(), 102);
    }

    #[test]
    fn test_calculate_tension() {
        // root_indices [2, 4, 6] → mean = 4.0
        let result = calculate_micro_tension(vec![2, 4, 6]).unwrap();
        assert!((result - 4.0).abs() < f64::EPSILON);
    }

    // -- edit_distance --

    #[test]
    fn test_edit_distance_identical() {
        assert_eq!(edit_distance("hello", "hello"), 0);
    }

    #[test]
    fn test_edit_distance_one_change() {
        assert_eq!(edit_distance("cat", "bat"), 1);
        assert_eq!(edit_distance("cat", "cats"), 1);
        assert_eq!(edit_distance("cat", "at"), 1);
    }

    #[test]
    fn test_edit_distance_empty() {
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", ""), 3);
        assert_eq!(edit_distance("", ""), 0);
    }

    // -- extract_character_names_regex --

    #[test]
    fn test_extract_names_dialogue_attribution() {
        let scenes = ["Alice said hello. Bob replied quickly. Alice asked why. \
             Alice whispered something. Bob said nothing. Bob muttered goodbye."];
        let result = extract_character_names_regex(&scenes).unwrap();
        let names: Vec<&str> = result.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"Alice"), "Should find Alice: {:?}", result);
        assert!(names.contains(&"Bob"), "Should find Bob: {:?}", result);
    }

    #[test]
    fn test_extract_names_filters_false_positives() {
        let scenes = ["The rain fell. Then it stopped. Then it started. Then more rain."];
        let result = extract_character_names_regex(&scenes).unwrap();
        let names: Vec<&str> = result.iter().map(|(n, _)| n.as_str()).collect();
        assert!(!names.contains(&"Then"), "Should filter 'Then'");
        assert!(!names.contains(&"The"), "Should filter 'The'");
    }

    #[test]
    fn test_extract_names_minimum_count() {
        let scenes = [
            "Then Marcus walked and Marcus ran and Marcus sat down.",
            "Only Zara appeared once.",
        ];
        let result = extract_character_names_regex(&scenes).unwrap();
        let names: Vec<&str> = result.iter().map(|(n, _)| n.as_str()).collect();
        assert!(
            names.contains(&"Marcus"),
            "Marcus appears 3 times: {:?}",
            result
        );
        assert!(!names.contains(&"Zara"), "Zara appears only once");
    }

    #[test]
    fn single_dialogue_attribution_keeps_name() {
        // A character with one dialogue attribution but few mentions used to be
        // dropped by the >=3 threshold; the strong-signal path now keeps it.
        let scenes = ["\"Hello there,\" said Eleanor as the door swung open."];
        let result = extract_character_names_regex(&scenes).unwrap();
        let names: Vec<&str> = result.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"Eleanor"), "got: {result:?}");
    }

    #[test]
    fn test_extract_names_empty() {
        let result = extract_character_names_regex(&[] as &[&str]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_names_filters_pronouns_and_connectives() {
        // Pronouns and connective openers used to surface as "characters" when
        // they recurred mid-clause. They must now be filtered while a real,
        // dialogue-attributed name in the same text survives.
        let scenes = [
            "He walked on. It was cold. He paused. It grew colder. He shivered. \
             As the wind rose, He turned. It howled. As before, He pressed on. \
             \"Onward,\" said Rowan, and He followed As It darkened.",
        ];
        let result = extract_character_names_regex(&scenes).unwrap();
        let names: Vec<&str> = result.iter().map(|(n, _)| n.as_str()).collect();
        for noise in ["He", "It", "As"] {
            assert!(
                !names.contains(&noise),
                "should filter '{noise}': {result:?}"
            );
        }
        assert!(names.contains(&"Rowan"), "real name kept: {result:?}");
    }

    #[test]
    fn test_is_plausible_name_rejects_caps_and_single_letter() {
        assert!(!is_plausible_name("CITY"));
        assert!(!is_plausible_name("A"));
        assert!(is_plausible_name("Mara"));
        assert!(is_plausible_name("Mary Smith"));
    }

    // -- extract_character_mentions --

    #[test]
    fn mentions_offsets_point_at_their_surface() {
        let scenes = ["Then Marcus walked and Marcus ran and Marcus sat down."];
        let mentions = extract_character_mentions(&scenes).unwrap();
        assert!(!mentions.is_empty(), "{mentions:?}");
        for mention in &mentions {
            let scene = scenes[mention.scene_index];
            let end = mention.byte_offset + mention.surface.len();
            assert_eq!(
                &scene[mention.byte_offset..end],
                mention.surface,
                "offset must slice back to the surface: {mention:?}"
            );
        }
    }

    #[test]
    fn mention_surfaces_match_name_extraction_survivors() {
        // Every mention surface is a survivor, and every survivor appears at least
        // once — the two extractors never disagree on which names are kept.
        let scenes = [
            "Then Marcus walked and Marcus ran and Marcus sat down.",
            "Only Zara appeared once.",
        ];
        let survivors: HashSet<String> = extract_character_names_regex(&scenes)
            .unwrap()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        let mention_surfaces: HashSet<String> = extract_character_mentions(&scenes)
            .unwrap()
            .into_iter()
            .map(|m| m.surface)
            .collect();
        assert_eq!(mention_surfaces, survivors, "kept-surface sets must match");
        assert!(!survivors.contains("Zara"), "single mention is dropped");
    }

    #[test]
    fn mentions_are_ordered_by_scene_then_offset() {
        let scenes = [
            "Marcus spoke. Marcus left. Marcus returned.",
            "Marcus arrived. Marcus waited. Marcus slept.",
        ];
        let mentions = extract_character_mentions(&scenes).unwrap();
        for pair in mentions.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            assert!(
                (a.scene_index, a.byte_offset) <= (b.scene_index, b.byte_offset),
                "not sorted: {a:?} then {b:?}"
            );
        }
    }

    #[test]
    fn mentions_empty_input_is_empty() {
        assert!(extract_character_mentions(&[] as &[&str])
            .unwrap()
            .is_empty());
    }

    // -- deduplicate_characters --

    #[test]
    fn test_deduplicate_substring() {
        let names = vec!["Elizabeth".into(), "Liz".into()];
        let result = deduplicate_characters(names).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "Elizabeth");
    }

    #[test]
    fn test_deduplicate_shared_first_word() {
        let names = vec!["Mary Smith".into(), "Mary".into()];
        let result = deduplicate_characters(names).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "Mary Smith");
    }

    #[test]
    fn test_deduplicate_distinct() {
        let names = vec!["Alice".into(), "Bob".into(), "Charlie".into()];
        let result = deduplicate_characters(names).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_deduplicate_edit_distance() {
        let names = vec!["Katherine".into(), "Katharine".into()];
        let result = deduplicate_characters(names).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_deduplicate_empty() {
        let result = deduplicate_characters(vec![]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_deduplicate_shared_last_word() {
        // Neither name is a substring of the other, so the shared-last-word
        // branch (which reads first/last words) decides the merge.
        let names = vec!["John Smith".into(), "Mary Smith".into()];
        let result = deduplicate_characters(names).unwrap();
        assert_eq!(result.len(), 1);
    }

    // -- merge_character_aliases --

    #[test]
    fn test_merge_aliases_substring() {
        let counts = vec![("Elizabeth".into(), 10), ("Liz".into(), 5)];
        let result = merge_character_aliases(counts).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "Elizabeth");
        assert_eq!(result[0].1, 15);
    }

    #[test]
    fn test_merge_aliases_shared_word() {
        let counts = vec![("Mary".into(), 8), ("Mary Smith".into(), 3)];
        let result = merge_character_aliases(counts).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 11);
    }

    #[test]
    fn test_merge_aliases_distinct() {
        let counts = vec![("Alice".into(), 10), ("Bob".into(), 5)];
        let result = merge_character_aliases(counts).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_merge_aliases_empty() {
        let result = merge_character_aliases(vec![]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_merge_aliases_shared_last_word() {
        // Neither name is a substring of the other; the shared-last-word branch
        // (reading first/last words) drives the merge.
        let counts = vec![("John Smith".into(), 7), ("Mary Smith".into(), 4)];
        let result = merge_character_aliases(counts).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 11);
    }

    #[test]
    fn test_merge_aliases_sorted_by_count() {
        let counts = vec![("Alice".into(), 5), ("Bob".into(), 10)];
        let result = merge_character_aliases(counts).unwrap();
        assert_eq!(result[0].0, "Bob");
        assert_eq!(result[1].0, "Alice");
    }

    // -- detect_themes_keywords --

    #[test]
    fn test_themes_basic() {
        let scenes = vec![vec![
            "love".into(),
            "heart".into(),
            "the".into(),
            "story".into(),
        ]];
        let result = detect_themes_keywords(scenes).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].contains_key("love"));
        assert!(result[0]["love"] > 0.0);
    }

    #[test]
    fn test_themes_no_match() {
        let scenes = vec![vec!["the".into(), "cat".into(), "sat".into()]];
        let result = detect_themes_keywords(scenes).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].is_empty());
    }

    #[test]
    fn test_themes_multiple_themes() {
        let scenes = vec![vec![
            "love".into(),
            "death".into(),
            "kill".into(),
            "heart".into(),
        ]];
        let result = detect_themes_keywords(scenes).unwrap();
        assert!(result[0].contains_key("love"));
        assert!(result[0].contains_key("death"));
    }

    #[test]
    fn test_themes_empty() {
        let result = detect_themes_keywords(vec![]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_themes_empty_scene() {
        let result = detect_themes_keywords(vec![vec![]]).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].is_empty());
    }

    // -- build_cooccurrence_edges --

    #[test]
    fn test_cooccurrence_basic() {
        let scenes = vec![
            vec!["alice".into(), "met".into(), "bob".into()],
            vec!["alice".into(), "met".into(), "bob".into()],
        ];
        let characters = vec!["alice".into(), "bob".into()];
        let result = build_cooccurrence_edges(scenes, characters).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].2, 2); // two scenes together
    }

    #[test]
    fn test_cooccurrence_no_overlap() {
        let scenes = vec![
            vec!["alice".into(), "walked".into()],
            vec!["bob".into(), "ran".into()],
        ];
        let characters = vec!["alice".into(), "bob".into()];
        let result = build_cooccurrence_edges(scenes, characters).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_cooccurrence_three_characters() {
        let scenes = vec![vec![
            "alice".into(),
            "bob".into(),
            "and".into(),
            "charlie".into(),
        ]];
        let characters = vec!["alice".into(), "bob".into(), "charlie".into()];
        let result = build_cooccurrence_edges(scenes, characters).unwrap();
        assert_eq!(result.len(), 3); // 3 pairs from 3 characters
    }

    #[test]
    fn test_cooccurrence_empty() {
        let result = build_cooccurrence_edges(vec![], vec!["alice".into()]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_cooccurrence_case_insensitive() {
        let scenes = vec![vec!["Alice".into(), "met".into(), "BOB".into()]];
        let characters = vec!["alice".into(), "bob".into()];
        let result = build_cooccurrence_edges(scenes, characters).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_cooccurrence_sorted_by_weight() {
        let scenes = vec![
            vec!["alice".into(), "bob".into()],
            vec!["alice".into(), "bob".into()],
            vec!["alice".into(), "charlie".into()],
        ];
        let characters = vec!["alice".into(), "bob".into(), "charlie".into()];
        let result = build_cooccurrence_edges(scenes, characters).unwrap();
        assert!(result[0].2 >= result[1].2);
    }

    // -- merge_character_aliases: count over length --

    #[test]
    fn test_merge_aliases_count_over_length() {
        // "Liz" has higher count than "Elizabeth"; canonical should be "Liz"
        let counts = vec![("Liz".into(), 20), ("Elizabeth".into(), 5)];
        let result = merge_character_aliases(counts).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "Liz", "Higher-count name should be canonical");
        assert_eq!(result[0].1, 25);
    }

    #[test]
    fn test_merge_aliases_shared_word_count_wins() {
        // "Mary" appears more than "Mary Smith"; canonical should be "Mary"
        let counts = vec![("Mary".into(), 15), ("Mary Smith".into(), 3)];
        let result = merge_character_aliases(counts).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "Mary", "Higher-count name should be canonical");
        assert_eq!(result[0].1, 18);
    }

    // -- false_positives: allow potential character names --

    #[test]
    fn test_false_positives_allows_one() {
        // "One" should not be filtered (could be a character name)
        let fp = false_positives();
        assert!(!fp.contains("one"), "'one' should not be a false positive");
    }

    #[test]
    fn test_false_positives_allows_just() {
        let fp = false_positives();
        assert!(
            !fp.contains("just"),
            "'just' should not be a false positive"
        );
    }

    #[test]
    fn test_false_positives_keeps_articles() {
        let fp = false_positives();
        assert!(fp.contains("the"), "'the' should remain a false positive");
        assert!(fp.contains("and"), "'and' should remain a false positive");
        assert!(fp.contains("but"), "'but' should remain a false positive");
        assert!(fp.contains("not"), "'not' should remain a false positive");
        assert!(
            fp.contains("chapter"),
            "'chapter' should remain a false positive"
        );
    }

    // -- agency ratio: lowercase comparison --

    #[test]
    fn test_agency_ratio_case_insensitive() {
        let relations = vec![
            ("verb".into(), "nsubj".into(), "alice".into()),
            ("verb".into(), "nsubj".into(), "ALICE".into()),
        ];
        let ratio = calculate_agency_ratio(relations, "Alice".into()).unwrap();
        assert!(
            ratio > 0.5,
            "Case-insensitive match should find both, got {ratio}"
        );
    }
}
