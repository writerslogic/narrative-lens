//! Word frequency lookups using a build-time generated sorted table of 50,000 English words.
//! Data sourced from the `wordfreq` Python package (COCA + subtitles + Wikipedia corpus).
//! Frequency lookups use binary search with zero runtime initialization cost.
//! Spelling suggestions lazily share a compact trie of the eligible lexicon.

// Include the generated frequency table (sorted arrays for binary search)
include!(concat!(env!("OUT_DIR"), "/freq_table.rs"));

/// Look up the Zipf frequency of an English word.
/// Returns 0.0 for unknown words.
/// Zipf scale: 7.0+ = "the", 6.0 = "is", 5.0 = "way", 4.0 = "actually", 3.0 = rare
pub fn zipf_frequency(word: &str) -> f32 {
    let lower = word.to_lowercase();
    match FREQ_WORDS.binary_search(&lower.as_str()) {
        Ok(idx) => FREQ_VALUES[idx],
        Err(_) => 0.0,
    }
}

const NO_NODE: u32 = u32::MAX;
const MAX_DISTANCE: usize = 2;

struct SpellingNode {
    character: char,
    child: u32,
    sibling: u32,
    word: u32,
}

struct SpellingIndex {
    nodes: Vec<SpellingNode>,
    max_length: usize,
}

impl SpellingIndex {
    fn build() -> Self {
        let mut index = Self {
            nodes: vec![SpellingNode {
                character: '\0',
                child: NO_NODE,
                sibling: NO_NODE,
                word: NO_NODE,
            }],
            max_length: 0,
        };
        for (word_index, word) in FREQ_WORDS.iter().enumerate() {
            if FREQ_VALUES[word_index] < 3.0 {
                continue;
            }
            let mut parent = 0;
            let mut length = 0;
            for character in word.chars() {
                length += 1;
                let mut child = index.nodes[parent].child;
                while child != NO_NODE && index.nodes[child as usize].character != character {
                    child = index.nodes[child as usize].sibling;
                }
                if child == NO_NODE {
                    child =
                        u32::try_from(index.nodes.len()).expect("embedded lexicon index fits u32");
                    index.nodes.push(SpellingNode {
                        character,
                        child: NO_NODE,
                        sibling: index.nodes[parent].child,
                        word: NO_NODE,
                    });
                    index.nodes[parent].child = child;
                }
                parent = child as usize;
            }
            index.nodes[parent].word =
                u32::try_from(word_index).expect("embedded lexicon index fits u32");
            index.max_length = index.max_length.max(length);
        }
        index.nodes.shrink_to_fit();
        index
    }

    fn search(
        &self,
        mut node: u32,
        query: &[char],
        previous: &[usize],
        rows: &mut [Vec<usize>],
        depth: usize,
        matches: &mut Vec<(usize, usize)>,
    ) {
        let Some((current, remaining_rows)) = rows.split_first_mut() else {
            return;
        };
        let lo = depth.saturating_sub(MAX_DISTANCE).max(1);
        let hi = (depth + MAX_DISTANCE).min(query.len());
        while node != NO_NODE {
            let entry = &self.nodes[node as usize];
            current[0] = depth;
            if lo > 1 {
                current[lo - 1] = MAX_DISTANCE + 1;
            }
            if hi < query.len() {
                current[hi + 1] = MAX_DISTANCE + 1;
            }
            let mut minimum = current[0];
            for column in lo..=hi {
                current[column] = (previous[column] + 1)
                    .min(current[column - 1] + 1)
                    .min(previous[column - 1] + usize::from(entry.character != query[column - 1]));
                minimum = minimum.min(current[column]);
            }
            if depth.abs_diff(query.len()) <= MAX_DISTANCE
                && current[query.len()] <= MAX_DISTANCE
                && entry.word != NO_NODE
            {
                matches.push((entry.word as usize, current[query.len()]));
            }
            if minimum <= MAX_DISTANCE {
                self.search(
                    entry.child,
                    query,
                    current,
                    remaining_rows,
                    depth + 1,
                    matches,
                );
            }
            node = entry.sibling;
        }
    }
}

/// Conservative Levenshtein-distance suggestions, ordered by distance, descending
/// frequency, then lexical order. Candidates retain the input's first character.
pub fn spelling_suggestions(word: &str, limit: usize) -> Vec<String> {
    if limit == 0 || word.is_empty() {
        return Vec::new();
    }
    static INDEX: std::sync::OnceLock<SpellingIndex> = std::sync::OnceLock::new();
    let index = INDEX.get_or_init(SpellingIndex::build);
    let length = word.chars().count();
    if length > index.max_length + MAX_DISTANCE {
        return Vec::new();
    }
    let query: Vec<char> = word.chars().collect();
    let mut first = index.nodes[0].child;
    while first != NO_NODE && index.nodes[first as usize].character != query[0] {
        first = index.nodes[first as usize].sibling;
    }
    if first == NO_NODE {
        return Vec::new();
    }
    let first = &index.nodes[first as usize];
    let suffix = &query[1..];
    let previous: Vec<usize> = (0..=suffix.len()).collect();
    let mut rows = vec![
        vec![MAX_DISTANCE + 1; suffix.len() + 1];
        (suffix.len() + MAX_DISTANCE).min(index.max_length.saturating_sub(1))
    ];
    let mut candidates = Vec::new();
    if suffix.len() <= MAX_DISTANCE && first.word != NO_NODE {
        candidates.push((first.word as usize, suffix.len()));
    }
    index.search(
        first.child,
        suffix,
        &previous,
        &mut rows,
        1,
        &mut candidates,
    );
    candidates.sort_by(|&(a, da), &(b, db)| {
        da.cmp(&db)
            .then_with(|| FREQ_VALUES[b].total_cmp(&FREQ_VALUES[a]))
            .then_with(|| FREQ_WORDS[a].cmp(FREQ_WORDS[b]))
    });
    candidates
        .into_iter()
        .take(limit)
        .map(|(index, _)| FREQ_WORDS[index].to_string())
        .collect()
}

#[cfg(test)]
fn edit_distance_bounded(a: &str, b: &str, maximum: usize) -> Option<usize> {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.len().abs_diff(b.len()) > maximum {
        return None;
    }
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut current = vec![i + 1; b.len() + 1];
        let mut row_min = current[0];
        for (j, cb) in b.iter().enumerate() {
            current[j + 1] = (previous[j + 1] + 1)
                .min(current[j] + 1)
                .min(previous[j] + usize::from(ca != cb));
            row_min = row_min.min(current[j + 1]);
        }
        if row_min > maximum {
            return None;
        }
        previous = current;
    }
    (previous[b.len()] <= maximum).then_some(previous[b.len()])
}

/// Compute vocabulary sophistication score (0-100).
/// Analyzes content words (4+ letters, alphabetic) and computes mean Zipf frequency.
/// Lower mean Zipf = rarer words = higher sophistication score.
/// Score = 100 - (mean_zipf - 2.0) * 25.0, clamped to [0, 100].
pub fn vocabulary_sophistication(tokens: Vec<String>) -> f64 {
    let content: Vec<&str> = tokens
        .iter()
        .filter(|t| t.len() >= 4 && t.chars().all(|c| c.is_alphabetic()))
        .map(|s| s.as_str())
        .collect();

    if content.is_empty() {
        return 50.0; // neutral default
    }

    let total_zipf: f64 = content.iter().map(|w| zipf_frequency(w) as f64).sum();
    let mean_zipf = total_zipf / content.len() as f64;

    // Transform: lower Zipf = more sophisticated
    (100.0 - (mean_zipf - 2.0) * 25.0).clamp(0.0, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_suggestions(word: &str) -> Vec<String> {
        let first = word.chars().next();
        let length = word.chars().count();
        let mut matches: Vec<_> = FREQ_WORDS
            .iter()
            .zip(FREQ_VALUES)
            .filter_map(|(&candidate, &frequency)| {
                if candidate.chars().next() != first
                    || candidate.chars().count().abs_diff(length) > MAX_DISTANCE
                    || frequency < 3.0
                {
                    return None;
                }
                edit_distance_bounded(word, candidate, MAX_DISTANCE)
                    .map(|distance| (candidate, distance, frequency))
            })
            .collect();
        matches.sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| b.2.total_cmp(&a.2))
                .then_with(|| a.0.cmp(b.0))
        });
        matches
            .into_iter()
            .map(|(word, _, _)| word.into())
            .collect()
    }

    #[test]
    fn component_performance_spelling_matches_scan_at_boundaries_and_ties() {
        let queries = [
            "teh",
            "recieve",
            "seperate",
            "acommodate",
            "definately",
            "occurence",
            "thier",
            "ca",
            "a",
            "café",
            "résumé",
            "éé",
            "The",
            "zzzzzzzzzzzzzzzzzz",
        ];
        let mut saw_frequency_tie = false;
        for query in queries {
            let expected = reference_suggestions(query);
            for limit in [0, 1, 3, usize::MAX] {
                assert_eq!(
                    spelling_suggestions(query, limit),
                    expected.iter().take(limit).cloned().collect::<Vec<_>>(),
                    "{query:?}, limit {limit}"
                );
            }
            for pair in expected.windows(2) {
                let a = edit_distance_bounded(query, &pair[0], MAX_DISTANCE).unwrap();
                let b = edit_distance_bounded(query, &pair[1], MAX_DISTANCE).unwrap();
                if a == b && zipf_frequency(&pair[0]) == zipf_frequency(&pair[1]) {
                    saw_frequency_tie = true;
                    assert!(pair[0] < pair[1]);
                }
            }
        }
        assert!(saw_frequency_tie);
        assert_eq!(edit_distance_bounded("cat", "cats", 2), Some(1));
        assert_eq!(edit_distance_bounded("cat", "catss", 2), Some(2));
        assert_eq!(edit_distance_bounded("cat", "catsss", 2), None);
        assert_eq!(edit_distance_bounded("teh", "the", 2), Some(2));
        assert!(spelling_suggestions("", 3).is_empty());
    }

    #[test]
    fn component_performance_spelling_matches_scan_for_dictionary_mutations() {
        for word in FREQ_WORDS
            .iter()
            .step_by(997)
            .filter(|w| w.len() >= 4)
            .take(20)
        {
            let chars: Vec<_> = word.chars().collect();
            let middle = chars.len() / 2;
            for operation in 0..4 {
                let mut mutated = chars.clone();
                match operation {
                    0 => {
                        mutated.remove(middle);
                    }
                    1 => mutated.insert(middle, 'q'),
                    2 => mutated[middle] = 'q',
                    _ => mutated.swap(middle - 1, middle),
                }
                let query: String = mutated.iter().collect();
                assert_eq!(
                    spelling_suggestions(&query, usize::MAX),
                    reference_suggestions(&query),
                    "{query:?}"
                );
            }
        }
        let longest = FREQ_WORDS
            .iter()
            .zip(FREQ_VALUES)
            .filter(|(_, frequency)| **frequency >= 3.0)
            .max_by_key(|(word, _)| word.chars().count())
            .unwrap()
            .0;
        for added in 1..=3 {
            let query = format!("{longest}{}", "q".repeat(added));
            assert_eq!(
                spelling_suggestions(&query, usize::MAX),
                reference_suggestions(&query)
            );
        }
    }

    #[test]
    fn test_common_word() {
        let z = zipf_frequency("the");
        assert!(z > 7.0, "Expected 'the' > 7.0, got {z}");
    }

    #[test]
    fn test_medium_word() {
        let z = zipf_frequency("beautiful");
        assert!(z > 4.0, "Expected 'beautiful' > 4.0, got {z}");
    }

    #[test]
    fn test_unknown_word() {
        assert_eq!(zipf_frequency("xyzzynonword"), 0.0);
    }

    #[test]
    fn test_case_insensitive() {
        assert_eq!(zipf_frequency("The"), zipf_frequency("the"));
    }

    #[test]
    fn test_sophistication_simple() {
        let score = vocabulary_sophistication(
            vec!["the", "quick", "brown", "good", "make"]
                .into_iter()
                .map(String::from)
                .collect(),
        );
        assert!(score < 80.0, "Simple words should score < 80, got {score}");
    }

    #[test]
    fn test_sophistication_empty() {
        assert_eq!(vocabulary_sophistication(vec![]), 50.0);
    }
}
