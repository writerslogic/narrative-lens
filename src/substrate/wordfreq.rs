//! Word frequency lookups using a build-time generated sorted table of 50,000 English words.
//! Data sourced from the `wordfreq` Python package (COCA + subtitles + Wikipedia corpus).
//! Uses binary search for O(log n) lookups with zero runtime initialization cost.

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

/// Conservative edit-distance suggestions from the embedded English lexicon.
pub fn spelling_suggestions(word: &str, limit: usize) -> Vec<String> {
    if limit == 0 || word.is_empty() {
        return Vec::new();
    }
    let first = word.chars().next();
    let length = word.chars().count();
    let mut candidates: Vec<(&str, usize, f32)> = FREQ_WORDS
        .iter()
        .zip(FREQ_VALUES.iter())
        .filter_map(|(&candidate, &frequency)| {
            let candidate_len = candidate.chars().count();
            if candidate.chars().next() != first
                || candidate_len.abs_diff(length) > 2
                || frequency < 3.0
            {
                return None;
            }
            let distance = edit_distance_bounded(word, candidate, 2)?;
            Some((candidate, distance, frequency))
        })
        .collect();
    candidates.sort_by(|a, b| {
        a.1.cmp(&b.1)
            .then_with(|| b.2.total_cmp(&a.2))
            .then_with(|| a.0.cmp(b.0))
    });
    candidates
        .into_iter()
        .take(limit)
        .map(|(word, _, _)| word.to_string())
        .collect()
}

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
