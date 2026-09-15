use std::collections::{HashMap, HashSet};

use log::debug;

use crate::substrate::wordfreq::zipf_frequency;

#[cfg(test)]
use crate::substrate::text::tokenize_words;

/// Adjectives that mark symbolic objects when used in "the [adj] [noun]" patterns.
const SYMBOLIC_ADJECTIVES: &[&str] = &[
    "old",
    "ancient",
    "strange",
    "peculiar",
    "mysterious",
    "forgotten",
    "hidden",
    "locked",
    "sealed",
];

/// Phrases that signal explicit foreshadowing promise language.
/// Checked via substring match on the raw scene text.
const PROMISE_PHRASES: &[&str] = &[
    "little did",
    "would later prove",
    "one day",
    "if only",
    "someday",
    "it would not be the last time",
    "would come to regret",
    "did not yet know",
];

/// Action verbs that indicate a foreshadowed item has been resolved.
const RESOLUTION_VERBS: &[&str] = &[
    "used",
    "opened",
    "revealed",
    "discovered",
    "broke",
    "destroyed",
    "fired",
    "grabbed",
    "wielded",
    "unlocked",
    "shattered",
    "activated",
    "drew",
    "seized",
];

/// Words that signal deliberate emphasis on an object or detail.
const EMPHASIS_WORDS: &[&str] = &[
    "noticed",
    "caught",
    "unusual",
    "strange",
    "peculiar",
    "odd",
    "remarkable",
    "glint",
    "glimmer",
    "glimpse",
    "eyed",
    "stared",
    "fixated",
    "conspicuous",
    "unmistakable",
];

/// Words that signal high narrative tension.
const TENSION_WORDS: &[&str] = &[
    "scream",
    "screamed",
    "blood",
    "death",
    "die",
    "died",
    "kill",
    "killed",
    "gun",
    "knife",
    "shatter",
    "shattered",
    "explode",
    "exploded",
    "crash",
    "crashed",
    "desperate",
    "panic",
    "terror",
    "horror",
    "fear",
    "dread",
    "threat",
    "danger",
    "trembled",
    "gasped",
    "fled",
    "attack",
    "attacked",
    "fought",
    "struggle",
    "collapsed",
    "fury",
    "rage",
];

#[derive(Clone, Debug)]
pub struct ForeshadowingItem {
    pub setup_term: String,
    pub setup_scene: usize,
    pub payoff_scene: Option<usize>,
    pub setup_count: usize,
    pub payoff_count: usize,
    pub payoff_quality: f64,
    pub status: String,
    pub suggestion: String,
}

/// Check if a word appears near any emphasis word within a window of tokens.
fn has_nearby_emphasis(tokens: &[String], word: &str, window: usize) -> bool {
    let emphasis: HashSet<&str> = EMPHASIS_WORDS.iter().copied().collect();
    for (i, tok) in tokens.iter().enumerate() {
        if tok == word {
            let start = i.saturating_sub(window);
            let end = (i + window + 1).min(tokens.len());
            for neighbor in &tokens[start..end] {
                if emphasis.contains(neighbor.as_str()) {
                    return true;
                }
            }
        }
    }
    false
}

/// Count how many tension words appear within a window around occurrences of `word`.
fn tension_density_near(tokens: &[String], word: &str, window: usize) -> usize {
    let tension: HashSet<&str> = TENSION_WORDS.iter().copied().collect();
    let mut count = 0;
    for (i, tok) in tokens.iter().enumerate() {
        if tok == word {
            let start = i.saturating_sub(window);
            let end = (i + window + 1).min(tokens.len());
            for neighbor in &tokens[start..end] {
                if tension.contains(neighbor.as_str()) {
                    count += 1;
                }
            }
        }
    }
    count
}

/// Detect symbolic objects: nouns preceded by symbolic adjectives in "the [adj] [noun]" patterns.
fn find_symbolic_objects(tokens: &[String]) -> HashSet<String> {
    let adj_set: HashSet<&str> = SYMBOLIC_ADJECTIVES.iter().copied().collect();
    let mut objects = HashSet::new();
    // Look for patterns: "the" + symbolic_adj + noun (3-token window)
    if tokens.len() < 3 {
        return objects;
    }
    for i in 0..tokens.len() - 2 {
        if tokens[i].to_lowercase() == "the"
            && adj_set.contains(tokens[i + 1].to_lowercase().as_str())
        {
            let noun = tokens[i + 2].to_lowercase();
            if noun.len() >= 3 && noun.chars().all(|c| c.is_alphabetic()) {
                objects.insert(noun);
            }
        }
    }
    objects
}

/// Check if a scene's raw text contains any promise/payoff foreshadowing phrases.
fn contains_promise_phrase(scene_lower: &str) -> bool {
    PROMISE_PHRASES
        .iter()
        .any(|phrase| scene_lower.contains(phrase))
}

/// Check if a word appears near resolution action verbs within a token window.
fn has_nearby_resolution_verb(tokens: &[String], word: &str, window: usize) -> bool {
    let verbs: HashSet<&str> = RESOLUTION_VERBS.iter().copied().collect();
    for (i, tok) in tokens.iter().enumerate() {
        if tok.to_lowercase() == word {
            let start = i.saturating_sub(window);
            let end = (i + window + 1).min(tokens.len());
            for neighbor in &tokens[start..end] {
                if verbs.contains(neighbor.to_lowercase().as_str()) {
                    return true;
                }
            }
        }
    }
    false
}

/// Analyze foreshadowing setup-payoff arcs across scenes.
///
/// `scenes_tokens`: pre-tokenized words for each scene.
/// `total_scenes`: total number of scenes (should equal scenes_tokens.len()).
///
/// Returns a list of `ForeshadowingItem` with quality scores and suggestions.
pub fn analyze_foreshadowing(
    scenes_tokens: Vec<Vec<String>>,
    total_scenes: usize,
) -> Vec<ForeshadowingItem> {
    debug!(
        "[foreshadow] analyze_foreshadowing — {} scenes",
        total_scenes
    );
    if total_scenes < 3 {
        return Vec::new();
    }

    let n = scenes_tokens.len().min(total_scenes);
    let first_third = n / 3;
    let last_third_start = n - n / 3;
    let climax_start = n - n.max(7) / 7; // last ~15%

    // Build per-third word occurrence sets and counts
    let mut setup_counts: HashMap<String, usize> = HashMap::new();
    let mut setup_first_scene: HashMap<String, usize> = HashMap::new();
    let mut middle_words: HashSet<String> = HashSet::new();
    let mut payoff_counts: HashMap<String, usize> = HashMap::new();
    let mut payoff_scenes: HashMap<String, Vec<usize>> = HashMap::new();

    // Track which words appear near emphasis in the setup
    let mut emphasized_words: HashSet<String> = HashSet::new();
    // Track symbolic objects ("the [adj] [noun]" patterns)
    let mut symbolic_objects: HashSet<String> = HashSet::new();
    // Track words near promise/payoff language
    let mut promise_words: HashSet<String> = HashSet::new();
    // Track structurally prominent words (first/last scene of each third)
    let mut structurally_prominent: HashSet<String> = HashSet::new();

    for (scene_idx, tokens) in scenes_tokens.iter().enumerate().take(n) {
        if scene_idx < first_third {
            // Setup third
            let mut seen_in_scene: HashSet<String> = HashSet::new();
            for tok in tokens {
                if tok.len() < 3 || !tok.chars().all(|c| c.is_alphabetic()) {
                    continue;
                }
                let lower = tok.to_lowercase();
                *setup_counts.entry(lower.clone()).or_insert(0) += 1;
                if !seen_in_scene.contains(&lower) {
                    seen_in_scene.insert(lower.clone());
                    setup_first_scene.entry(lower.clone()).or_insert(scene_idx);
                }
            }
            // Check emphasis
            for word in &seen_in_scene {
                if has_nearby_emphasis(tokens, word, 5) {
                    emphasized_words.insert(word.clone());
                }
            }
            // Symbolic object detection
            for obj in find_symbolic_objects(tokens) {
                symbolic_objects.insert(obj);
            }
            // Promise/payoff language: mark all nouns in scenes with promise phrases
            let scene_text: String = tokens.join(" ").to_lowercase();
            if contains_promise_phrase(&scene_text) {
                for word in &seen_in_scene {
                    promise_words.insert(word.clone());
                }
            }
            // Structurally prominent: first and last scene of setup
            if scene_idx == 0 || scene_idx == first_third.saturating_sub(1) {
                for word in &seen_in_scene {
                    structurally_prominent.insert(word.clone());
                }
            }
        } else if scene_idx < last_third_start {
            // Middle third
            for tok in tokens {
                if tok.len() < 3 || !tok.chars().all(|c| c.is_alphabetic()) {
                    continue;
                }
                middle_words.insert(tok.to_lowercase());
            }
        } else {
            // Payoff third
            let mut seen_in_scene: HashSet<String> = HashSet::new();
            for tok in tokens {
                if tok.len() < 3 || !tok.chars().all(|c| c.is_alphabetic()) {
                    continue;
                }
                let lower = tok.to_lowercase();
                *payoff_counts.entry(lower.clone()).or_insert(0) += 1;
                if !seen_in_scene.contains(&lower) {
                    seen_in_scene.insert(lower.clone());
                    payoff_scenes
                        .entry(lower.clone())
                        .or_default()
                        .push(scene_idx);
                }
            }
        }
    }

    // Identify candidate foreshadowing terms:
    // Pattern 1: appears in setup, absent from middle, AND reappears in payoff
    //   (a genuine setup→payoff arc — both halves required, per this comment).
    // Pattern 2: appears near emphasis words in setup (regardless of middle)
    // Pattern 3: symbolic object ("the [adj] [noun]" with foreshadowing adjectives)
    // Pattern 4: appears in scene with promise/payoff language
    // Filter: must have Zipf < 5.0 (not too common)
    let mut candidates: HashSet<String> = HashSet::new();

    for word in setup_counts.keys() {
        let z = zipf_frequency(word);
        if z >= 5.0 {
            continue; // too common
        }
        let absent_from_middle = !middle_words.contains(word);
        let reappears_in_payoff = payoff_counts.contains_key(word);
        let is_emphasized = emphasized_words.contains(word);
        let is_symbolic = symbolic_objects.contains(word);

        // Pattern 1 requires BOTH absence-from-middle and a payoff reappearance.
        // The earlier code admitted a word on absence-from-middle ALONE, so on a
        // real novel essentially every moderately-rare setup word that happened
        // not to recur in the middle third became a "motif" (thousands of nodes).
        //
        // Promise language is deliberately NOT a standalone admission signal: the
        // promise detector marks *every* word in any scene containing a phrase
        // like "little did she know", so on a real manuscript a handful of such
        // scenes promote hundreds of unrelated words to motifs. Promise proximity
        // instead boosts the payoff *quality* of words admitted on structural
        // grounds (see the `promise_words` bonus below). Emphasis and symbolic
        // structure remain standalone signals — both are word-local (a windowed
        // emphasis marker; a "the [adj] [noun]" pattern), so they mark a specific
        // deliberate setup rather than a whole scene's vocabulary.
        let pattern_1 = absent_from_middle && reappears_in_payoff;
        if pattern_1 || is_emphasized || is_symbolic {
            candidates.insert(word.clone());
        }
    }

    // Score each candidate
    let mut results: Vec<ForeshadowingItem> = Vec::new();

    for term in &candidates {
        let s_count = *setup_counts.get(term).unwrap_or(&0);
        let p_count = *payoff_counts.get(term).unwrap_or(&0);
        let first_scene = *setup_first_scene.get(term).unwrap_or(&0);
        let p_scenes = payoff_scenes.get(term);

        let last_payoff = p_scenes.and_then(|v| v.last().copied());
        let payoff_scene_count = p_scenes.map(|v| v.len()).unwrap_or(0);

        // Compute payoff quality
        let quality = if p_count == 0 {
            0.0
        } else {
            let mut q: f64 = 0.3; // base for any reappearance

            // Bonus for appearing in multiple payoff scenes
            if payoff_scene_count >= 2 {
                q += 0.2;
            }

            // Bonus for appearing in climax (last 15%)
            let in_climax = p_scenes
                .map(|v| v.iter().any(|&s| s >= climax_start))
                .unwrap_or(false);
            if in_climax {
                q += 0.25;
            }

            // Bonus for tension context in payoff scenes
            let mut total_tension = 0;
            for (scene_idx, tokens) in scenes_tokens.iter().enumerate().take(n) {
                if scene_idx >= last_third_start {
                    total_tension += tension_density_near(tokens, term, 8);
                }
            }
            if total_tension >= 2 {
                q += 0.15;
            } else if total_tension >= 1 {
                q += 0.08;
            }

            // Improved resolution: bonus if term appears with action verbs in final 25%
            let final_quarter_start = n - n / 4;
            let has_resolution = scenes_tokens
                .iter()
                .enumerate()
                .take(n)
                .any(|(idx, tokens)| {
                    idx >= final_quarter_start && has_nearby_resolution_verb(tokens, term, 6)
                });
            if has_resolution {
                q += 0.15;
            }

            // Bonus for symbolic objects and promise-language terms
            if symbolic_objects.contains(term) {
                q += 0.1;
            }
            if promise_words.contains(term) {
                q += 0.1;
            }

            // Penalty for being only a single passing mention
            if p_count == 1 && payoff_scene_count == 1 && !in_climax {
                q = q.min(0.5);
            }

            q.clamp(0.0, 1.0)
        };

        // Status
        let status = if s_count > 5 {
            "overcrowded"
        } else if quality >= 0.8 {
            "resolved_strong"
        } else if quality >= 0.2 {
            "resolved_weak"
        } else if p_count == 0 {
            "unresolved"
        } else {
            "resolved_weak"
        };

        // Suggestion
        let suggestion = match status {
            "unresolved" => format!(
                "'{}' is introduced in scene {} but never pays off. Either remove the setup or add a payoff in Act 3.",
                term, first_scene
            ),
            "resolved_weak" => format!(
                "'{}' reappears but only briefly. Strengthen the callback by connecting it to the climax.",
                term
            ),
            "overcrowded" => format!(
                "'{}' is mentioned {} times in the setup. This may telegraph too heavily. Consider reducing to 2-3 mentions.",
                term, s_count
            ),
            "resolved_strong" => format!(
                "'{}' has a strong setup-payoff arc from scene {} to scene {}.",
                term,
                first_scene,
                last_payoff.unwrap_or(first_scene)
            ),
            _ => String::new(),
        };

        results.push(ForeshadowingItem {
            setup_term: term.clone(),
            setup_scene: first_scene,
            payoff_scene: last_payoff,
            setup_count: s_count,
            payoff_count: p_count,
            payoff_quality: quality,
            status: status.to_string(),
            suggestion,
        });
    }

    // Sort by quality descending, then by setup_scene ascending
    results.sort_by(|a, b| {
        b.payoff_quality
            .partial_cmp(&a.payoff_quality)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.setup_scene.cmp(&b.setup_scene))
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scene(words: &str) -> Vec<String> {
        tokenize_words(words)
    }

    #[test]
    fn test_too_few_scenes() {
        let scenes = vec![make_scene("hello world")];
        let result = analyze_foreshadowing(scenes, 1);
        assert!(result.is_empty());
    }

    #[test]
    fn test_classic_chekhov_gun() {
        // 9 scenes: setup (0-2), middle (3-5), payoff (6-8)
        let scenes = vec![
            // Setup: introduce "revolver" near emphasis
            make_scene("She noticed the peculiar revolver hanging on the wall"),
            make_scene("The old revolver gleamed in morning light"),
            make_scene("She wondered about the revolver again"),
            // Middle: no mention of revolver
            make_scene("They traveled across the countryside"),
            make_scene("The storm gathered over the hills"),
            make_scene("Arguments erupted at dinner"),
            // Payoff: revolver returns with tension
            make_scene("He grabbed the revolver from the wall"),
            make_scene("The revolver exploded with a deafening crash"),
            make_scene("The revolver fell to the floor as panic spread"),
        ];
        let result = analyze_foreshadowing(scenes, 9);
        let revolver = result.iter().find(|r| r.setup_term == "revolver");
        assert!(
            revolver.is_some(),
            "Should detect 'revolver' as foreshadowing"
        );
        let r = revolver.unwrap();
        assert!(
            r.payoff_quality >= 0.5,
            "Revolver should score at least moderate, got {}",
            r.payoff_quality
        );
        assert!(r.payoff_count > 0);
        assert!(r.payoff_scene.is_some());
    }

    #[test]
    fn test_unresolved_foreshadowing() {
        // 9 scenes: locket appears in setup only
        let scenes = vec![
            make_scene("She clutched the strange locket around her neck"),
            make_scene("The locket felt warm against her skin"),
            make_scene("She tucked the locket away"),
            make_scene("They marched through the forest"),
            make_scene("Rain fell on the weary travelers"),
            make_scene("Camp was set by the river"),
            make_scene("Dawn broke over the mountains"),
            make_scene("They reached the castle gates"),
            make_scene("The battle began at noon"),
        ];
        let result = analyze_foreshadowing(scenes, 9);
        let locket = result.iter().find(|r| r.setup_term == "locket");
        assert!(locket.is_some(), "Should detect 'locket' as unresolved");
        let l = locket.unwrap();
        assert_eq!(l.payoff_quality, 0.0);
        assert_eq!(l.status, "unresolved");
        assert!(l.suggestion.contains("never pays off"));
    }

    #[test]
    fn test_overcrowded() {
        // Term mentioned 6+ times in setup
        let scenes = vec![
            make_scene("The dagger was sharp. The dagger gleamed. She held the dagger tightly. The dagger was ancient. Another dagger reference. The dagger shone."),
            make_scene("The dagger again"),
            make_scene("More about the dagger here"),
            make_scene("Walking in the garden"),
            make_scene("Nothing happened"),
            make_scene("Still nothing"),
            make_scene("The dagger appeared once more"),
            make_scene("She used the dagger"),
            make_scene("The dagger ended it all"),
        ];
        let result = analyze_foreshadowing(scenes, 9);
        let dagger = result.iter().find(|r| r.setup_term == "dagger");
        assert!(dagger.is_some(), "Should detect 'dagger'");
        let d = dagger.unwrap();
        assert_eq!(d.status, "overcrowded");
        assert!(d.suggestion.contains("telegraph too heavily"));
    }

    #[test]
    fn test_common_words_filtered() {
        // Common words (Zipf >= 5.0) should not be candidates
        let scenes = vec![
            make_scene("The door opened and the hand reached out"),
            make_scene("Another day with the door"),
            make_scene("The door was still there"),
            make_scene("Walking through the forest"),
            make_scene("Nothing special"),
            make_scene("Still nothing"),
            make_scene("The door opened again"),
            make_scene("She reached for the door"),
            make_scene("The door closed"),
        ];
        let result = analyze_foreshadowing(scenes, 9);
        // "door" and "hand" are common words, should be filtered
        let _door = result.iter().find(|r| r.setup_term == "door");
        // Depending on Zipf score, "door" may or may not be filtered.
        // But truly common words like "the", "and" should never appear.
        let the = result.iter().find(|r| r.setup_term == "the");
        assert!(the.is_none(), "'the' should be filtered as too common");
    }

    #[test]
    fn test_emphasis_detection() {
        // Word near emphasis markers should be detected even if it appears in middle
        let scenes = vec![
            make_scene("She noticed the peculiar medallion on the shelf"),
            make_scene("It was an ordinary morning"),
            make_scene("The sun rose slowly"),
            make_scene("The medallion sat there unnoticed"),
            make_scene("Days passed uneventfully"),
            make_scene("More time passed"),
            make_scene("The medallion was the key to everything"),
            make_scene("She grabbed the medallion"),
            make_scene("The medallion unlocked the gate with a crash"),
        ];
        let result = analyze_foreshadowing(scenes, 9);
        let medallion = result.iter().find(|r| r.setup_term == "medallion");
        assert!(
            medallion.is_some(),
            "Should detect 'medallion' via emphasis pattern"
        );
    }

    #[test]
    fn test_symbolic_object_detection() {
        // "the ancient chalice" should be detected as a symbolic object
        let scenes = vec![
            make_scene("On the mantle sat the ancient chalice covered in dust"),
            make_scene("The chalice gleamed when light struck it"),
            make_scene("She left the room thinking of the chalice"),
            make_scene("The journey continued through the plains"),
            make_scene("Nothing remarkable happened"),
            make_scene("They rested at camp"),
            make_scene("She grabbed the chalice and used it to open the door"),
            make_scene("The chalice revealed a hidden passage"),
            make_scene("The chalice shattered as the spell broke"),
        ];
        let result = analyze_foreshadowing(scenes, 9);
        let chalice = result.iter().find(|r| r.setup_term == "chalice");
        assert!(
            chalice.is_some(),
            "Should detect 'chalice' as symbolic object"
        );
        let c = chalice.unwrap();
        assert!(c.payoff_quality > 0.0, "Chalice should have payoff quality");
    }

    #[test]
    fn test_promise_phrase_detection() {
        // Promise language "little did she know" should boost detection
        let scenes = vec![
            make_scene(
                "She picked up the talisman. Little did she know it would change everything",
            ),
            make_scene("The talisman was warm to the touch"),
            make_scene("She carried the talisman in her pocket"),
            make_scene("The road was long and winding"),
            make_scene("They stopped at an inn"),
            make_scene("The night was uneventful"),
            make_scene("The talisman began to glow"),
            make_scene("She used the talisman against the darkness"),
            make_scene("The talisman destroyed the barrier"),
        ];
        let result = analyze_foreshadowing(scenes, 9);
        let talisman = result.iter().find(|r| r.setup_term == "talisman");
        assert!(
            talisman.is_some(),
            "Should detect 'talisman' via promise phrase"
        );
    }

    #[test]
    fn test_resolution_verb_bonus() {
        // Term reappearing near action verbs in final quarter gets bonus
        let scenes = vec![
            make_scene("The strange amulet hung on the wall"),
            make_scene("She eyed the amulet with curiosity"),
            make_scene("The amulet was forgotten"),
            make_scene("The village celebrated a festival"),
            make_scene("Days turned into weeks"),
            make_scene("Nothing changed in the village"),
            make_scene("She remembered the amulet"),
            make_scene("She grabbed the amulet and destroyed the seal"),
            make_scene("The amulet revealed the hidden truth at last"),
        ];
        let result = analyze_foreshadowing(scenes, 9);
        let amulet = result.iter().find(|r| r.setup_term == "amulet");
        assert!(amulet.is_some(), "Should detect 'amulet'");
        let a = amulet.unwrap();
        assert!(
            a.payoff_quality >= 0.5,
            "Amulet with resolution verbs should score well, got {}",
            a.payoff_quality
        );
    }
}
