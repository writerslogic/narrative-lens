use aho_corasick::AhoCorasick;
use regex::Regex;
use std::sync::OnceLock;

use crate::craft::lexicon::WordMatcher;

use log::debug;

/// A prose quality issue found in a scene.
#[derive(Clone, Debug)]
pub struct ProseExample {
    pub scene_id: usize,
    pub issue_type: String,
    pub snippet: String,
    pub suggestion: String,
}

const MAX_PROSE_EXAMPLES: usize = 25;

fn adverb_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b\w+ly\b").unwrap())
}

const NON_ADVERBS: &[&str] = &[
    "only", "early", "likely", "family", "holy", "ugly", "lonely",
];

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..crate::substrate::text::floor_char_boundary(s, max)])
    }
}

// ---------------------------------------------------------------------------
// Cliche detection
// ---------------------------------------------------------------------------

const CLICHE_PHRASES: &[&str] = &[
    "dark as night",
    "quiet as a mouse",
    "light as a feather",
    "cold as ice",
    "heart skipped a beat",
    "blood ran cold",
    "time stood still",
    "nod slowly",
    "nodded slowly",
    "let out a breath",
    "tears streamed down",
    "in the nick of time",
    "against all odds",
    "it was a dark and stormy night",
    "piercing blue eyes",
    "raven black hair",
    "chiseled jaw",
    "breathe a sigh of relief",
    "sent shivers down",
    "knuckles turned white",
    "pit of her stomach",
    "pit of his stomach",
    "eyes widened in shock",
    "deafening silence",
    "crystal clear",
    "better late than never",
    "at the end of the day",
    "sigh of relief",
    "stomach dropped",
    "bated breath",
    "the last straw",
    "tip of the iceberg",
];

struct ClicheMatcher {
    automaton: AhoCorasick,
}

fn cliche_matcher() -> &'static ClicheMatcher {
    static INST: OnceLock<ClicheMatcher> = OnceLock::new();
    INST.get_or_init(|| {
        let automaton = AhoCorasick::new(CLICHE_PHRASES).expect("cliche automaton");
        ClicheMatcher { automaton }
    })
}

// ---------------------------------------------------------------------------
// Filter words (narrative distance)
// ---------------------------------------------------------------------------

const FILTER_WORDS: &[&str] = &[
    "seemed", "felt", "realized", "noticed", "watched", "heard", "saw", "thought", "wondered",
    "knew",
];

fn filter_word_matcher() -> &'static WordMatcher {
    static M: OnceLock<WordMatcher> = OnceLock::new();
    M.get_or_init(|| WordMatcher::new(FILTER_WORDS))
}

/// Returns true if the sentence looks like dialogue (starts/ends with quotes).
fn is_dialogue(sent: &str) -> bool {
    let t = sent.trim();
    t.starts_with('"')
        || t.starts_with('\u{201c}')
        || t.starts_with('\u{2018}')
        || t.starts_with('\'')
}

// ---------------------------------------------------------------------------
// Purple prose: 3+ consecutive adjectives
// ---------------------------------------------------------------------------

const COMMON_ADJECTIVES: &[&str] = &[
    "big",
    "small",
    "old",
    "young",
    "new",
    "long",
    "short",
    "tall",
    "dark",
    "bright",
    "cold",
    "warm",
    "hot",
    "cool",
    "soft",
    "hard",
    "sharp",
    "smooth",
    "rough",
    "thick",
    "thin",
    "deep",
    "shallow",
    "wide",
    "narrow",
    "heavy",
    "light",
    "fast",
    "slow",
    "loud",
    "quiet",
    "dry",
    "wet",
    "clean",
    "dirty",
    "rich",
    "poor",
    "sweet",
    "bitter",
    "sour",
    "ancient",
    "vast",
    "tiny",
    "massive",
    "pale",
    "faint",
    "gentle",
    "fierce",
    "wild",
    "calm",
    "broken",
    "silent",
    "golden",
    "silver",
    "crimson",
    "scarlet",
    "azure",
    "emerald",
    "ivory",
    "obsidian",
    "velvet",
    "gossamer",
    "luminous",
    "shadowy",
    "ethereal",
    "delicate",
    "ornate",
    "twisted",
    "gnarled",
    "withered",
    "blooming",
    "radiant",
    "somber",
    "dreary",
    "lush",
    "barren",
    "pristine",
    "tattered",
    "weathered",
    "gleaming",
];

/// Collect prose quality issues from scenes: overlong sentences, passive voice,
/// adverb clusters, repetitive openers, cliches, floating-head dialogue,
/// filter words, monotonous rhythm, and purple prose.
/// Also includes show-don't-tell and word echo results from crate::substrate::text.
pub fn collect_prose_examples(scenes: &[String]) -> Vec<ProseExample> {
    debug!(
        "[prose_quality] collect_prose_examples — {} scenes",
        scenes.len()
    );
    let mut examples = Vec::new();
    let passive = crate::substrate::text::passive_regex();
    let adverb = adverb_re();
    let cliche = cliche_matcher();
    let filter_matcher = filter_word_matcher();

    for (i, scene_text) in scenes.iter().enumerate() {
        if examples.len() >= MAX_PROSE_EXAMPLES {
            break;
        }

        // Abbreviation-aware sentence split via the shared primitive.
        let sents = crate::substrate::text::split_sentences(scene_text);

        let mut prev_opener = String::new();
        let mut repeat_count = 0u32;
        let mut sent_word_counts: Vec<usize> = Vec::new();

        for sent in &sents {
            if examples.len() >= MAX_PROSE_EXAMPLES {
                break;
            }
            let words: Vec<&str> = sent.split_whitespace().collect();
            let wc = words.len();
            sent_word_counts.push(wc);

            // Overlong sentences
            if wc > 50 {
                examples.push(ProseExample {
                    scene_id: i,
                    issue_type: "overlong_sentence".into(),
                    snippet: truncate(sent, 120),
                    suggestion: format!(
                        "This {}-word sentence may lose readers. Consider splitting it.",
                        wc
                    ),
                });
            }

            // Passive voice
            if passive.is_match(sent) && wc > 8 {
                examples.push(ProseExample {
                    scene_id: i,
                    issue_type: "passive_voice".into(),
                    snippet: truncate(sent, 120),
                    suggestion: "Consider rewriting in active voice for stronger impact.".into(),
                });
            }

            // Adverb clusters
            let adverbs: Vec<String> = adverb
                .find_iter(sent)
                .map(|m| m.as_str().to_string())
                .filter(|a| !NON_ADVERBS.contains(&a.to_lowercase().as_str()))
                .collect();
            if adverbs.len() >= 3 {
                let shown: Vec<&str> = adverbs.iter().take(3).map(|s| s.as_str()).collect();
                examples.push(ProseExample {
                    scene_id: i,
                    issue_type: "adverb_cluster".into(),
                    snippet: truncate(sent, 120),
                    suggestion: format!(
                        "Found {} adverbs ({}). Consider removing some.",
                        adverbs.len(),
                        shown.join(", ")
                    ),
                });
            }

            // Repetitive openers
            let opener = words.first().map(|w| w.to_lowercase()).unwrap_or_default();
            if !opener.is_empty() && opener == prev_opener {
                repeat_count += 1;
                if repeat_count >= 2 {
                    examples.push(ProseExample {
                        scene_id: i,
                        issue_type: "repetitive_opener".into(),
                        snippet: truncate(sent, 120),
                        suggestion: format!(
                            "Multiple consecutive sentences starting with '{}'. Vary your sentence openings.",
                            words[0]
                        ),
                    });
                    repeat_count = 0;
                }
            } else {
                repeat_count = 0;
            }
            prev_opener = opener;

            // Filter words in narration
            if !is_dialogue(sent)
                && wc > 5
                && let Some(m) = filter_matcher.find_first(sent)
            {
                examples.push(ProseExample {
                    scene_id: i,
                    issue_type: "filter_word".into(),
                    snippet: truncate(sent, 120),
                    suggestion: format!(
                        "Filter word '{}' creates narrative distance. Describe directly instead.",
                        m.text.to_lowercase()
                    ),
                });
            }

            // Purple prose: 3+ adjectives modifying the same noun
            if wc >= 4 {
                let lower_words: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();
                let adj_set: std::collections::HashSet<&str> =
                    COMMON_ADJECTIVES.iter().copied().collect();
                let mut consecutive_adj = 0usize;
                for lw in &lower_words {
                    let cleaned = lw.trim_matches(|c: char| c.is_ascii_punctuation());
                    if adj_set.contains(cleaned) {
                        consecutive_adj += 1;
                    } else {
                        if consecutive_adj >= 3 {
                            examples.push(ProseExample {
                                scene_id: i,
                                issue_type: "purple_prose".into(),
                                snippet: truncate(sent, 120),
                                suggestion: format!(
                                    "Found {} consecutive adjectives. Trim to 1-2 for stronger prose.",
                                    consecutive_adj
                                ),
                            });
                            break;
                        }
                        consecutive_adj = 0;
                    }
                }
                // Check at end of sentence too
                if consecutive_adj >= 3
                    && !examples.iter().any(|e| {
                        e.scene_id == i
                            && e.issue_type == "purple_prose"
                            && e.snippet == truncate(sent, 120)
                    })
                {
                    examples.push(ProseExample {
                        scene_id: i,
                        issue_type: "purple_prose".into(),
                        snippet: truncate(sent, 120),
                        suggestion: format!(
                            "Found {} consecutive adjectives. Trim to 1-2 for stronger prose.",
                            consecutive_adj
                        ),
                    });
                }
            }
        }

        // Cliche detection (whole scene)
        let lower_scene = scene_text.to_lowercase();
        for mat in cliche.automaton.find_iter(&lower_scene) {
            if examples.len() >= MAX_PROSE_EXAMPLES {
                break;
            }
            let matched = &lower_scene[mat.start()..mat.end()];
            let ctx_start =
                crate::substrate::text::floor_char_boundary(scene_text, mat.start().saturating_sub(20));
            let ctx_end = crate::substrate::text::ceil_char_boundary(
                scene_text,
                (mat.end() + 20).min(scene_text.len()),
            );
            let context = &scene_text[ctx_start..ctx_end];
            examples.push(ProseExample {
                scene_id: i,
                issue_type: "cliche".into(),
                snippet: truncate(context, 120),
                suggestion: format!(
                    "Cliche '{}' weakens the prose. Find a fresh, specific image.",
                    matched
                ),
            });
        }

        // Dialogue without action beats: 5+ consecutive quoted lines
        detect_floating_head(scene_text, i, &mut examples);

        // Sentence variety: 4+ consecutive sentences with similar length (within 20%)
        if sent_word_counts.len() >= 4 {
            let mut run_start = 0;
            for j in 1..sent_word_counts.len() {
                let prev = sent_word_counts[j - 1] as f64;
                let curr = sent_word_counts[j] as f64;
                let similar = if prev > 0.0 && curr > 0.0 {
                    let ratio = curr / prev;
                    (0.8..=1.2).contains(&ratio)
                } else {
                    prev == curr
                };
                if !similar {
                    let run_len = j - run_start;
                    if run_len >= 4 && examples.len() < MAX_PROSE_EXAMPLES {
                        let avg_len =
                            sent_word_counts[run_start..j].iter().sum::<usize>() / run_len;
                        examples.push(ProseExample {
                            scene_id: i,
                            issue_type: "monotonous_rhythm".into(),
                            snippet: truncate(sents[run_start], 120),
                            suggestion: format!(
                                "{} consecutive sentences average ~{} words each. Vary sentence length for better rhythm.",
                                run_len, avg_len
                            ),
                        });
                    }
                    run_start = j;
                }
            }
            // Check final run
            let run_len = sent_word_counts.len() - run_start;
            if run_len >= 4 && examples.len() < MAX_PROSE_EXAMPLES {
                let avg_len = sent_word_counts[run_start..].iter().sum::<usize>() / run_len;
                examples.push(ProseExample {
                    scene_id: i,
                    issue_type: "monotonous_rhythm".into(),
                    snippet: truncate(sents[run_start], 120),
                    suggestion: format!(
                        "{} consecutive sentences average ~{} words each. Vary sentence length for better rhythm.",
                        run_len, avg_len
                    ),
                });
            }
        }
    }

    // Also collect show-don't-tell and word echo issues from crate::substrate::text
    for (i, scene_text) in scenes.iter().enumerate() {
        if examples.len() >= MAX_PROSE_EXAMPLES * 2 {
            break;
        }
        let sdt = crate::substrate::text::detect_show_dont_tell(scene_text, i);
        for issue in sdt {
            examples.push(ProseExample {
                scene_id: issue.scene_id,
                issue_type: issue.issue_type,
                snippet: issue.snippet,
                suggestion: issue.suggestion,
            });
        }
        let echoes = crate::substrate::text::detect_word_echoes(scene_text, i);
        for issue in echoes {
            examples.push(ProseExample {
                scene_id: issue.scene_id,
                issue_type: issue.issue_type,
                snippet: issue.snippet,
                suggestion: issue.suggestion,
            });
        }
    }

    examples
}

/// Detect "floating head" dialogue: 5+ consecutive lines that are quoted speech
/// without intervening narration or action beats.
fn detect_floating_head(scene_text: &str, scene_id: usize, examples: &mut Vec<ProseExample>) {
    let lines: Vec<&str> = scene_text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    let mut consecutive_dialogue = 0usize;
    let mut run_start = 0usize;
    for (idx, line) in lines.iter().enumerate() {
        if is_dialogue(line) {
            if consecutive_dialogue == 0 {
                run_start = idx;
            }
            consecutive_dialogue += 1;
        } else {
            if consecutive_dialogue >= 5 && examples.len() < MAX_PROSE_EXAMPLES {
                examples.push(ProseExample {
                    scene_id,
                    issue_type: "floating_head_dialogue".into(),
                    snippet: truncate(lines[run_start], 120),
                    suggestion: format!(
                        "{} consecutive dialogue lines without action beats. Add gestures, movement, or sensory detail between lines.",
                        consecutive_dialogue
                    ),
                });
            }
            consecutive_dialogue = 0;
        }
    }
    // Check trailing run
    if consecutive_dialogue >= 5 && examples.len() < MAX_PROSE_EXAMPLES {
        examples.push(ProseExample {
            scene_id,
            issue_type: "floating_head_dialogue".into(),
            snippet: truncate(lines[run_start], 120),
            suggestion: format!(
                "{} consecutive dialogue lines without action beats. Add gestures, movement, or sensory detail between lines.",
                consecutive_dialogue
            ),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert!(collect_prose_examples(&[]).is_empty());
    }

    #[test]
    fn test_overlong_sentence() {
        let long = (0..55).map(|_| "word").collect::<Vec<_>>().join(" ") + ".";
        let results = collect_prose_examples(&[long]);
        assert!(results.iter().any(|p| p.issue_type == "overlong_sentence"));
    }

    #[test]
    fn test_passive_voice() {
        let text =
            "The report was completed by the team after a long and careful review.".to_string();
        let results = collect_prose_examples(&[text]);
        assert!(results.iter().any(|p| p.issue_type == "passive_voice"));
    }

    #[test]
    fn test_adverb_cluster() {
        let text = "She quickly gracefully silently moved through the door.".to_string();
        let results = collect_prose_examples(&[text]);
        assert!(results.iter().any(|p| p.issue_type == "adverb_cluster"));
    }

    #[test]
    fn test_repetitive_opener() {
        let text = "She ran. She jumped. She fell. She cried.".to_string();
        let results = collect_prose_examples(&[text]);
        assert!(results.iter().any(|p| p.issue_type == "repetitive_opener"));
    }

    #[test]
    fn test_max_examples_cap() {
        let scenes: Vec<String> = (0..50)
            .map(|_| {
                (0..55).map(|_| "word").collect::<Vec<_>>().join(" ")
                    + ". "
                    + &(0..55).map(|_| "word").collect::<Vec<_>>().join(" ")
                    + "."
            })
            .collect();
        let results = collect_prose_examples(&scenes);
        // Should be capped, not unlimited
        assert!(results.len() <= MAX_PROSE_EXAMPLES * 3);
    }

    #[test]
    fn test_non_adverbs_filtered() {
        let text = "The only family likely arrived early.".to_string();
        let results = collect_prose_examples(&[text]);
        assert!(!results.iter().any(|p| p.issue_type == "adverb_cluster"));
    }

    #[test]
    fn test_cliche_detection() {
        let text = "Her heart skipped a beat when she saw him.".to_string();
        let results = collect_prose_examples(&[text]);
        assert!(results.iter().any(|p| p.issue_type == "cliche"));
    }

    #[test]
    fn test_cliche_multiple() {
        let text = "It was a dark and stormy night. His blood ran cold.".to_string();
        let results = collect_prose_examples(&[text]);
        let cliches: Vec<_> = results
            .iter()
            .filter(|p| p.issue_type == "cliche")
            .collect();
        assert!(cliches.len() >= 2);
    }

    #[test]
    fn test_no_cliche_in_clean_prose() {
        let text = "The morning light filtered through the curtains.".to_string();
        let results = collect_prose_examples(&[text]);
        assert!(!results.iter().any(|p| p.issue_type == "cliche"));
    }

    #[test]
    fn test_floating_head_dialogue() {
        let text = "\"Hello,\" she said.\n\"Hi there.\"\n\"How are you?\"\n\"Fine.\"\n\"Good.\"\n\"Great.\"".to_string();
        let results = collect_prose_examples(&[text]);
        assert!(results
            .iter()
            .any(|p| p.issue_type == "floating_head_dialogue"));
    }

    #[test]
    fn test_no_floating_head_with_beats() {
        let text = "\"Hello,\" she said.\nHe shifted his weight.\n\"Hi there.\"\nShe smiled.\n\"How are you?\"".to_string();
        let results = collect_prose_examples(&[text]);
        assert!(!results
            .iter()
            .any(|p| p.issue_type == "floating_head_dialogue"));
    }

    #[test]
    fn test_filter_word_detection() {
        let text = "He noticed the door was slightly open and wondered about it.".to_string();
        let results = collect_prose_examples(&[text]);
        assert!(results.iter().any(|p| p.issue_type == "filter_word"));
    }

    #[test]
    fn test_filter_word_not_in_dialogue() {
        let text = "\"She seemed upset,\" he said.".to_string();
        let results = collect_prose_examples(&[text]);
        assert!(!results.iter().any(|p| p.issue_type == "filter_word"));
    }

    #[test]
    fn test_monotonous_rhythm() {
        // 6 sentences all around 8-9 words (within 20%)
        let text = "The cat sat on the warm mat. The dog lay on the soft rug. The bird sang on the old branch. The fish swam in the cold stream. The mouse hid in the dark hole. The frog sat on the wet rock.".to_string();
        let results = collect_prose_examples(&[text]);
        assert!(results.iter().any(|p| p.issue_type == "monotonous_rhythm"));
    }

    #[test]
    fn test_no_monotony_with_varied_lengths() {
        let text = "Stop. The cat sat on the warm mat by the window overlooking the garden. Run. She laughed.".to_string();
        let results = collect_prose_examples(&[text]);
        assert!(!results.iter().any(|p| p.issue_type == "monotonous_rhythm"));
    }

    #[test]
    fn test_purple_prose_detection() {
        let text = "The dark ancient twisted gnarled tree stood alone.".to_string();
        let results = collect_prose_examples(&[text]);
        assert!(results.iter().any(|p| p.issue_type == "purple_prose"));
    }

    #[test]
    fn test_no_purple_with_few_adjectives() {
        let text = "The old tree stood alone.".to_string();
        let results = collect_prose_examples(&[text]);
        assert!(!results.iter().any(|p| p.issue_type == "purple_prose"));
    }
}
