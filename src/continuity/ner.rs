use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use crate::onnx;

/// Set a custom directory to search for NER model files.
/// Must be called before the first call to `ner_extract_entities`.
pub fn set_ner_model_directory(path: &str) {
    onnx::set_ner_model_directory(path);
}

/// The BERT NER model accepts at most 512 tokens; its tokenizer has truncation
/// disabled, so feeding a longer text makes the ONNX run fail and silently
/// return nothing. English averages ~4 chars/token, so a window well under that
/// limit keeps every chunk valid with margin for subword splits.
const NER_WINDOW_CHARS: usize = 1500;

/// Run BERT NER over `text`, returning (entity_text, label, start_char,
/// end_char) in the **full-text** coordinate frame.
///
/// Long inputs are split into ≤[`NER_WINDOW_CHARS`] windows on whitespace (so a
/// name is never cut mid-token) and each window is tagged separately, with
/// offsets shifted back. This is what lets NER evaluate an entire scene: the
/// earlier fixed-char cap dropped every entity past the model's token limit,
/// leaving long scenes with no person/place evidence at all.
///
/// Delegates to the active [`inference`] backend (ONNX Runtime natively,
/// `tract` on wasm); returns an empty Vec when no model is available.
pub fn ner_extract_entities(text: &str) -> Vec<(String, String, usize, usize)> {
    ner_extract_entities_batch(&[text])
        .into_iter()
        .next()
        .unwrap_or_default()
}

/// How many NER windows to feed the model per padded ONNX run. Windows are
/// near-uniform (~[`NER_WINDOW_CHARS`]), so padding waste within a batch is
/// small while one matmul replaces this many sequential inferences.
const NER_BATCH: usize = 16;

/// Tag entities across many texts with as few padded ONNX runs as possible.
///
/// Every text is split into ≤[`NER_WINDOW_CHARS`] windows, the windows from all
/// texts are pooled, run through the model [`NER_BATCH`] at a time, and the
/// spans are regrouped per text with offsets shifted back to that text's frame.
/// This turns the old one-inference-per-window cost (minutes on a real
/// manuscript) into a handful of batched runs. Returns one entity list per
/// input text, in order.
pub fn ner_extract_entities_batch(texts: &[&str]) -> Vec<Vec<(String, String, usize, usize)>> {
    // window k belongs to text `text_idx` and starts at byte `start` within it.
    struct Win {
        text_idx: usize,
        start: usize,
    }
    let mut wins: Vec<Win> = Vec::new();
    let mut win_strs: Vec<&str> = Vec::new();
    for (ti, text) in texts.iter().enumerate() {
        if text.len() <= NER_WINDOW_CHARS {
            wins.push(Win {
                text_idx: ti,
                start: 0,
            });
            win_strs.push(text);
        } else {
            for (start, end) in ner_windows(text, NER_WINDOW_CHARS) {
                wins.push(Win {
                    text_idx: ti,
                    start,
                });
                win_strs.push(&text[start..end]);
            }
        }
    }

    let mut out: Vec<Vec<(String, String, usize, usize)>> = vec![Vec::new(); texts.len()];
    if win_strs.is_empty() {
        return out;
    }
    for base in (0..win_strs.len()).step_by(NER_BATCH) {
        let end = (base + NER_BATCH).min(win_strs.len());
        for (k, ents) in onnx::tag_batch_cached(&win_strs[base..end])
            .into_iter()
            .enumerate()
        {
            let w = &wins[base + k];
            for (name, label, s, e) in ents {
                out[w.text_idx].push((name, label, w.start + s, w.start + e));
            }
        }
    }
    out
}

/// Partition `text` into covering windows of at most `max` bytes, breaking on
/// whitespace where possible and always on a char boundary. Guarantees forward
/// progress so the loop terminates even on a single over-long token.
fn ner_windows(text: &str, max: usize) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = crate::substrate::text::floor_char_boundary(text, (start + max).min(text.len()));
        if end < text.len() {
            // Back up to the last whitespace so an entity isn't split across windows.
            if let Some(ws) = text[start..end].rfind(char::is_whitespace)
                && ws > 0
            {
                end = start + ws + 1;
            }
        }
        if end <= start {
            end = crate::substrate::text::ceil_char_boundary(text, start + 1);
        }
        spans.push((start, end));
        start = end;
    }
    spans
}

// ---------------------------------------------------------------------------
// 5-Phase Self-Supervised Character Extraction
// ---------------------------------------------------------------------------

fn capitalized_name_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?")
            .expect("capitalized-name regex")
    })
}

/// Phase 1 output: the per-name signal counts plus the set of surfaces the NER
/// tagger confidently labeled as a *non-person* entity (LOC/ORG/MISC) and never
/// as a person. These are used downstream to veto place/organization names
/// ("New York", "Associates") that the capitalization/regex signals would
/// otherwise admit as characters.
struct Phase1 {
    candidates: HashMap<String, [usize; 3]>,
    /// Title-cased surfaces (and their component words) NER labeled non-person.
    non_person: HashSet<String>,
    /// Title-cased surfaces with dialogue attribution ("X said") — high-precision
    /// person evidence that protects a name from the place/common-word vetoes.
    protected: HashSet<String>,
}

/// Phase 1: Broad candidate collection from all signal sources.
fn phase1_collect(scenes: &[String]) -> Phase1 {
    let mut candidates: HashMap<String, [usize; 3]> = HashMap::new();
    // NER label tallies per surface: how often it was a person vs. not.
    let mut per_hits: HashMap<String, usize> = HashMap::new();
    let mut non_per_hits: HashMap<String, usize> = HashMap::new();

    // Signal A: BERT NER. Collect PER as a positive signal and LOC/ORG/MISC as
    // negative evidence (a place/org is not a character even when capitalized).
    // Every scene is windowed and the windows from all scenes are tagged in
    // pooled batched ONNX runs — so the tagger sees every occurrence (no
    // fixed-length cap that would blind it deeper in a long scene) without
    // paying one inference per window.
    let scene_refs: Vec<&str> = scenes.iter().map(String::as_str).collect();
    for scene_entities in ner_extract_entities_batch(&scene_refs) {
        for (name, label, _, _) in scene_entities {
            let norm = title_case(name.trim());
            if norm.is_empty() {
                continue;
            }
            if label == "PER" {
                candidates.entry(norm.clone()).or_insert([0, 0, 0])[0] += 1;
                *per_hits.entry(norm).or_insert(0) += 1;
            } else {
                *non_per_hits.entry(norm).or_insert(0) += 1;
            }
        }
    }

    // A surface is a confident non-person only if NER labeled it non-person and
    // never a person. Add the full surface and its component words so split
    // fragments ("York" from "New York") are vetoed too.
    let mut non_person: HashSet<String> = HashSet::new();
    for (surface, &non_per) in &non_per_hits {
        if non_per > 0 && per_hits.get(surface).copied().unwrap_or(0) == 0 {
            non_person.insert(surface.clone());
            for word in surface.split_whitespace() {
                if word.len() >= 3 {
                    non_person.insert(title_case(word));
                }
            }
        }
    }

    // Signal B: Dialogue attribution regex (from nlp.rs)
    if let Ok(regex_results) = crate::substrate::nlp::extract_character_names_regex(scenes) {
        for (name, count) in regex_results {
            let norm = title_case(&name);
            candidates.entry(norm).or_insert([0, 0, 0])[1] += count;
        }
    }

    // Signal C: Mid-sentence capitalization (simple scan). Sentence-start
    // offsets come from the shared abbreviation-aware segmenter, not a regex.
    let cap_re = capitalized_name_re();
    for scene in scenes {
        let starts: HashSet<usize> = crate::substrate::utils::sentence_spans(scene)
            .into_iter()
            .map(|(s, _)| s)
            .collect();
        for m in cap_re.find_iter(scene) {
            if !starts.contains(&m.start()) {
                let name = m.as_str().to_string();
                let norm = title_case(&name);
                candidates.entry(norm).or_insert([0, 0, 0])[2] += 1;
            }
        }
    }

    let protected: HashSet<String> = crate::substrate::nlp::dialogue_attributed_names(scenes)
        .into_iter()
        .map(|n| title_case(&n))
        .collect();

    Phase1 {
        candidates,
        non_person,
        protected,
    }
}

/// Phase 2: Establish anchors (names with 2+ signal agreement AND 3+ total mentions).
fn phase2_anchors(candidates: &HashMap<String, [usize; 3]>) -> HashSet<String> {
    let mut anchors = HashSet::new();
    for (name, counts) in candidates {
        let signals_present = counts.iter().filter(|&&c| c > 0).count();
        let total: usize = counts.iter().sum();
        if signals_present >= 2 && total >= 3 {
            anchors.insert(name.clone());
        }
    }
    anchors
}

/// Phase 3: Learn naming patterns from anchors.
struct LearnedPatterns {
    /// Title prefixes observed (e.g., "Lord", "Dr.", "Professor")
    title_prefixes: HashSet<String>,
    /// Common name endings in this manuscript (for fantasy name detection)
    name_suffixes: HashSet<String>,
}

fn phase3_learn(anchors: &HashSet<String>, scenes: &[String]) -> LearnedPatterns {
    let known_titles: HashSet<&str> = [
        "lord",
        "lady",
        "sir",
        "dame",
        "king",
        "queen",
        "prince",
        "princess",
        "captain",
        "colonel",
        "general",
        "major",
        "sergeant",
        "commander",
        "doctor",
        "dr",
        "professor",
        "prof",
        "father",
        "mother",
        "brother",
        "sister",
        "master",
        "mistress",
        "count",
        "countess",
        "duke",
        "duchess",
        "baron",
        "uncle",
        "aunt",
        "grandma",
        "grandpa",
        "elder",
    ]
    .into_iter()
    .collect();

    let mut title_prefixes = HashSet::new();
    let mut suffixes: HashMap<String, usize> = HashMap::new();

    // Scan scenes for title + anchor name patterns
    for scene in scenes {
        let lower = scene.to_lowercase();
        for anchor in anchors {
            let anchor_lower = anchor.to_lowercase();
            // Check for title preceding this anchor
            if let Some(pos) = lower.find(&anchor_lower) {
                if pos >= 2 {
                    // Look at the word before the name
                    let before = lower[..pos].trim_end();
                    if let Some(last_word) = before.split_whitespace().last() {
                        let clean = last_word.trim_matches(|c: char| !c.is_alphabetic());
                        if known_titles.contains(clean) {
                            title_prefixes.insert(title_case(clean));
                        }
                    }
                }
            }

            // Collect name suffixes (last 3 chars) for pattern learning
            if anchor.len() >= 4 {
                let suffix = &anchor[crate::substrate::text::floor_char_boundary(anchor, anchor.len() - 3)..];
                *suffixes.entry(suffix.to_lowercase()).or_insert(0) += 1;
            }
        }
    }

    // Only keep suffixes that appear in 2+ anchors (real pattern, not noise)
    let name_suffixes: HashSet<String> = suffixes
        .into_iter()
        .filter(|(_, count)| *count >= 2)
        .map(|(s, _)| s)
        .collect();

    LearnedPatterns {
        title_prefixes,
        name_suffixes,
    }
}

/// Phase 4: Extended extraction using learned patterns.
fn phase4_extend(
    candidates: &HashMap<String, [usize; 3]>,
    anchors: &HashSet<String>,
    patterns: &LearnedPatterns,
    scenes: &[String],
) -> HashSet<String> {
    let mut extended = anchors.clone();

    // 4a: Promote single-signal candidates that match learned patterns
    for (name, counts) in candidates {
        if extended.contains(name) {
            continue;
        }
        let total: usize = counts.iter().sum();
        if total < 2 {
            continue;
        }

        let mut confidence = 0.0f64;

        // NER signal is strong
        if counts[0] > 0 {
            confidence += 0.4;
        }
        // Dialogue attribution is strong
        if counts[1] > 0 {
            confidence += 0.35;
        }
        // Capitalization alone is weak
        if counts[2] > 0 {
            confidence += 0.15;
        }

        // Bonus: matches a learned suffix pattern. Take the last 3 *characters*,
        // not bytes — a name ending in a multibyte char (accented letter, or a
        // stray Windows-1252 control) must not be sliced mid-character.
        if name.chars().count() >= 4 {
            let start = crate::substrate::text::floor_char_boundary(name, name.len() - 3);
            let suffix = name[start..].to_lowercase();
            if patterns.name_suffixes.contains(&suffix) {
                confidence += 0.15;
            }
        }

        // Bonus: high frequency suggests real character
        if total >= 5 {
            confidence += 0.1;
        }
        if total >= 10 {
            confidence += 0.1;
        }

        if confidence >= 0.5 {
            extended.insert(name.clone());
        }
    }

    // 4b: Find title + name patterns in text that weren't in candidates
    if !patterns.title_prefixes.is_empty() {
        let title_pat = patterns
            .title_prefixes
            .iter()
            .map(|t| regex::escape(t))
            .collect::<Vec<_>>()
            .join("|");
        if let Ok(re) = regex::Regex::new(&format!(r"(?:{})\s+([A-Z][a-z]{{2,}})", title_pat)) {
            for scene in scenes {
                for cap in re.captures_iter(scene) {
                    if cap.get(1).is_some() {
                        let full = cap.get(0).unwrap().as_str().to_string();
                        let norm = title_case(&full);
                        if !extended.contains(&norm) {
                            extended.insert(norm);
                        }
                    }
                }
            }
        }
    }

    extended
}

/// Phase 5: Structural validation.
fn phase5_validate(
    characters: &HashSet<String>,
    candidates: &HashMap<String, [usize; 3]>,
    non_person: &HashSet<String>,
    protected: &HashSet<String>,
) -> Vec<(String, usize)> {
    let mut result: Vec<(String, usize)> = Vec::new();

    for name in characters {
        let counts = candidates.get(name).copied().unwrap_or([0, 0, 0]);
        let total: usize = counts.iter().sum();

        // Must appear at least twice total
        if total < 2 {
            continue;
        }

        // Must not be a single very short word (likely noise)
        if name.len() < 3 {
            continue;
        }

        // A name has person evidence if NER ever tagged it PER, or it appears in
        // a dialogue-attribution frame ("Paris said") — a place never speaks.
        // Such names are exempt from the place/common-word vetoes, so a character
        // *named* Paris or Hope survives even when NER misses the person.
        let has_person_evidence = counts[0] > 0 || protected.contains(name);

        // NER veto: a surface NER confidently labeled a place/org/misc (and never
        // a person) is not a character, even when capitalized in prose.
        if !has_person_evidence && non_person.contains(name) {
            continue;
        }

        // Common-word veto: an ordinary high-frequency English word ("Just",
        // "City", "Change") capitalized mid-sentence is almost never a name.
        if !has_person_evidence && is_common_word(name) {
            continue;
        }

        result.push((name.clone(), total));
    }

    // Merge aliases using existing Levenshtein-based merger
    let merged = match crate::substrate::nlp::merge_character_aliases(result) {
        Ok(m) => m,
        Err(_) => return Vec::new(),
    };

    let mut final_result: Vec<(String, usize)> =
        merged.into_iter().filter(|(_, c)| *c >= 2).collect();
    final_result.sort_by_key(|x| std::cmp::Reverse(x.1));
    final_result
}

/// 5-Phase self-supervised character extraction.
/// Combines BERT NER, dialogue attribution, capitalization patterns,
/// anchor-based pattern learning, and structural validation.
pub fn ner_extract_persons(scenes: Vec<String>) -> Vec<(String, usize)> {
    if scenes.is_empty() {
        return Vec::new();
    }

    // Phase 1: Broad collection from all signals
    let Phase1 {
        candidates,
        non_person,
        protected,
    } = phase1_collect(&scenes);
    if candidates.is_empty() {
        return Vec::new();
    }

    // Phase 2: Establish high-confidence anchors
    let anchors = phase2_anchors(&candidates);

    // Phase 3: Learn manuscript-specific naming patterns
    let patterns = phase3_learn(&anchors, &scenes);

    // Phase 4: Extended extraction using learned patterns
    let extended = phase4_extend(&candidates, &anchors, &patterns, &scenes);

    // Phase 5: Structural validation and alias merging
    phase5_validate(&extended, &candidates, &non_person, &protected)
}

/// Whether a single word is a common English word (Zipf ≥ 4.5 ≈ top few
/// thousand words). Such words ("Just", "City", "Life", "New") are vetoed as
/// character names unless NER positively tags them as a person. Multi-word
/// surfaces are never treated as common words.
fn is_common_word(name: &str) -> bool {
    !name.contains(' ') && crate::substrate::wordfreq::zipf_frequency(name) >= 4.5
}

fn title_case(s: &str) -> String {
    s.split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => {
                    let upper: String = first.to_uppercase().collect();
                    upper + &chars.as_str().to_lowercase()
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_title_case() {
        assert_eq!(title_case("john doe"), "John Doe");
        assert_eq!(title_case("ALICE"), "Alice");
        assert_eq!(title_case(""), "");
    }

    #[test]
    fn test_phase2_anchors() {
        let mut cands = HashMap::new();
        cands.insert("Alice".to_string(), [5, 3, 2]); // all 3 signals
        cands.insert("Bob".to_string(), [2, 0, 1]); // 2 signals, 3 total
        cands.insert("Noise".to_string(), [0, 0, 1]); // 1 signal
        let anchors = phase2_anchors(&cands);
        assert!(anchors.contains("Alice"));
        assert!(anchors.contains("Bob"));
        assert!(!anchors.contains("Noise"));
    }

    #[test]
    fn test_phase5_filters_short() {
        let mut chars = HashSet::new();
        chars.insert("Al".to_string());
        chars.insert("Alice".to_string());
        let mut cands = HashMap::new();
        cands.insert("Al".to_string(), [1, 0, 0]);
        cands.insert("Alice".to_string(), [3, 2, 1]);
        let result = phase5_validate(&chars, &cands, &HashSet::new(), &HashSet::new());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "Alice");
    }

    #[test]
    fn test_phase5_vetoes_non_person_and_common_words() {
        // "Paris" is NER-tagged a place (in non_person, no PER signal) and "City"
        // is a common word with no PER signal — both must be dropped. "Marcus"
        // (real name, capitalization-only) and "Just" *with* a PER signal survive.
        let mut chars = HashSet::new();
        for n in ["Paris", "City", "Marcus", "Just"] {
            chars.insert(n.to_string());
        }
        let mut cands = HashMap::new();
        cands.insert("Paris".to_string(), [0, 2, 3]); // place, no PER
        cands.insert("City".to_string(), [0, 2, 2]); // common word, no PER
        cands.insert("Marcus".to_string(), [0, 0, 4]); // real name, cap-only
        cands.insert("Just".to_string(), [3, 0, 2]); // common word but PER-tagged
        let non_person = HashSet::from(["Paris".to_string()]);
        let result = phase5_validate(&chars, &cands, &non_person, &HashSet::new());
        let names: HashSet<&str> = result.iter().map(|(n, _)| n.as_str()).collect();
        assert!(!names.contains("Paris"), "place must be vetoed");
        assert!(!names.contains("City"), "common word must be vetoed");
        assert!(names.contains("Marcus"), "real name must survive");
        assert!(
            names.contains("Just"),
            "PER-tagged word overrides common-word veto"
        );
    }

    #[test]
    fn test_character_named_paris_survives_via_dialogue() {
        // A character literally named "Paris" whom NER never tags PER (it only
        // ever saw the city sense) must still survive because she speaks:
        // dialogue attribution puts her in `protected`.
        let mut chars = HashSet::new();
        chars.insert("Paris".to_string());
        let mut cands = HashMap::new();
        cands.insert("Paris".to_string(), [0, 4, 3]); // no PER signal
        let non_person = HashSet::from(["Paris".to_string()]); // NER called it a place
        let protected = HashSet::from(["Paris".to_string()]); // but "Paris said ..."
        let result = phase5_validate(&chars, &cands, &non_person, &protected);
        let names: HashSet<&str> = result.iter().map(|(n, _)| n.as_str()).collect();
        assert!(
            names.contains("Paris"),
            "dialogue attribution must protect a character named after a place"
        );
    }

    #[test]
    fn test_ner_windows_cover_and_respect_boundaries() {
        // Multibyte text longer than the window: windows must tile the whole
        // string contiguously, never exceed the cap, and split only on char
        // boundaries (em-dashes/smart quotes must never panic or be cut).
        let text = "Mr. Dalloway walked—slowly—toward Paris. ".repeat(80);
        let max = 100;
        let spans = ner_windows(&text, max);
        assert_eq!(spans[0].0, 0);
        assert_eq!(spans.last().unwrap().1, text.len());
        for (i, &(s, e)) in spans.iter().enumerate() {
            assert!(e > s, "window must make progress");
            assert!(e - s <= max, "window within cap");
            assert!(text.is_char_boundary(s) && text.is_char_boundary(e));
            if i > 0 {
                assert_eq!(spans[i - 1].1, s, "windows must be contiguous");
            }
        }
    }

    #[test]
    fn test_graceful_no_model() {
        let result = ner_extract_persons(vec!["John went to Paris.".to_string()]);
        let _ = result;
    }
}
