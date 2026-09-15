use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HookType {
    Action,
    Question,
    Voice,
    Situation,
    Character,
    Image,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpeningAnalysisResult {
    pub hook_type: Option<HookType>,
    pub hook_score: f64,
    pub first_line_quality: f64,
    pub first_paragraph_quality: f64,
    pub backstory_ratio: f64,
    pub voice_establishment: f64,
    pub engagement_trajectory: Vec<f64>,
    pub promises_made: Vec<String>,
    pub characters_introduced: usize,
    pub world_established: bool,
    pub genre_alignment: f64,
    pub read_on_prediction: f64,
    /// Word count of the first line (structural signal).
    pub first_line_word_count: usize,
    /// Word counts per paragraph (structural signal: paragraph-length trajectory).
    pub paragraph_word_counts: Vec<usize>,
    /// Index of the first paragraph that contains dialogue (0-based), or None.
    pub dialogue_onset_paragraph: Option<usize>,
    pub issues: Vec<String>,
    pub strengths: Vec<String>,
    pub agent_perspective: String,
    pub first_line_feedback: String,
    pub revision_suggestion: String,
}

/// Backstory/exposition indicators.
const BACKSTORY_MARKERS: &[&str] = &[
    "had been",
    "had always",
    "used to",
    "years ago",
    "back when",
    "growing up",
    "as a child",
    "remembered when",
    "it all started",
    "the history of",
    "was born in",
    "for generations",
    "long before",
    "in the old days",
    "once upon a time",
    "ever since",
    "from the beginning",
    "originally",
];

/// Action hook indicators (used for hook classification only).
const ACTION_INDICATORS: &[&str] = &[
    "ran",
    "jumped",
    "crashed",
    "exploded",
    "screamed",
    "slammed",
    "fired",
    "dodged",
    "sprinted",
    "dove",
    "lunged",
    "shattered",
];

/// Question/mystery indicators.
const QUESTION_INDICATORS: &[&str] = &[
    "?",
    "who",
    "why",
    "what if",
    "wondered",
    "mystery",
    "strange",
    "impossible",
    "shouldn't have",
    "disappeared",
];

/// Voice/character indicators.
const VOICE_INDICATORS: &[&str] = &["i ", "my ", "me ", "we ", "our "];

/// Situation/setting indicators.
const SITUATION_INDICATORS: &[&str] = &[
    "the city",
    "the town",
    "the world",
    "the year was",
    "in the",
    "on the",
    "at the",
    "when the",
];

/// Sensory/image indicators (used for hook classification and world detection only).
const IMAGE_INDICATORS: &[&str] = &[
    "smell", "taste", "sound", "light", "dark", "cold", "hot", "wind", "rain", "blood", "silence",
    "shadow",
];

/// Genre-specific opening quality modifiers.
fn genre_opening_preferences(genre: &str) -> (Vec<&'static str>, f64) {
    match genre.to_lowercase().as_str() {
        "thriller" => (vec!["action", "question", "situation"], 0.9),
        "romance" => (vec!["character", "voice", "situation"], 0.8),
        "mystery" => (vec!["question", "situation", "image"], 0.9),
        "literary" | "literary fiction" => (vec!["voice", "image", "character"], 0.85),
        "fantasy" => (vec!["situation", "image", "action"], 0.8),
        "sci-fi" | "science fiction" => (vec!["situation", "question", "image"], 0.85),
        "horror" => (vec!["image", "situation", "question"], 0.9),
        "historical" => (vec!["situation", "image", "voice"], 0.8),
        "ya" | "young adult" => (vec!["voice", "character", "action"], 0.85),
        _ => (vec!["voice", "action", "question"], 0.75),
    }
}

fn classify_hook(first_line: &str, first_paragraph: &str) -> (Option<HookType>, f64) {
    let line_lower = first_line.to_lowercase();
    let para_lower = first_paragraph.to_lowercase();

    let mut scores: Vec<(HookType, f64)> = Vec::new();

    // Action
    let action_count = ACTION_INDICATORS
        .iter()
        .filter(|w| line_lower.contains(*w))
        .count();
    if action_count > 0 {
        scores.push((HookType::Action, 0.5 + action_count as f64 * 0.15));
    }

    // Question
    let question_count = QUESTION_INDICATORS
        .iter()
        .filter(|w| para_lower.contains(*w))
        .count();
    if question_count > 0 || first_line.contains('?') {
        let bonus = if first_line.contains('?') { 0.2 } else { 0.0 };
        scores.push((
            HookType::Question,
            0.4 + question_count as f64 * 0.1 + bonus,
        ));
    }

    // Voice (first person, distinctive)
    let voice_count = VOICE_INDICATORS
        .iter()
        .filter(|w| line_lower.starts_with(*w) || line_lower.contains(*w))
        .count();
    if voice_count > 0 {
        scores.push((HookType::Voice, 0.3 + voice_count as f64 * 0.15));
    }

    // Situation
    let situation_count = SITUATION_INDICATORS
        .iter()
        .filter(|w| para_lower.contains(*w))
        .count();
    if situation_count > 0 {
        scores.push((HookType::Situation, 0.3 + situation_count as f64 * 0.1));
    }

    // Image
    let image_count = IMAGE_INDICATORS
        .iter()
        .filter(|w| para_lower.contains(*w))
        .count();
    if image_count > 0 {
        scores.push((HookType::Image, 0.4 + image_count as f64 * 0.12));
    }

    // Character (proper nouns approximation: capitalized words not at start of sentence)
    let words: Vec<&str> = first_paragraph.split_whitespace().collect();
    let char_signals = words
        .iter()
        .skip(1)
        .filter(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) && w.len() > 1)
        .count();
    if char_signals > 1 {
        scores.push((
            HookType::Character,
            0.3 + (char_signals as f64 * 0.08).min(0.4),
        ));
    }

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if let Some((hook_type, score)) = scores.into_iter().next() {
        (Some(hook_type), score.min(1.0))
    } else {
        (None, 0.2)
    }
}

fn calculate_backstory_ratio(text: &str) -> f64 {
    let sentences: Vec<&str> = text
        .split(['.', '!', '?'])
        .filter(|s| !s.trim().is_empty())
        .collect();

    if sentences.is_empty() {
        return 0.0;
    }

    let backstory_sentences = sentences
        .iter()
        .filter(|s| {
            let lower = s.to_lowercase();
            BACKSTORY_MARKERS
                .iter()
                .any(|marker| lower.contains(marker))
        })
        .count();

    backstory_sentences as f64 / sentences.len() as f64
}

fn detect_promises(text: &str) -> Vec<String> {
    let mut promises = Vec::new();
    let lower = text.to_lowercase();

    if lower.contains("secret") || lower.contains("hidden") {
        promises.push("A secret to be revealed".to_string());
    }
    if lower.contains("danger") || lower.contains("threat") || lower.contains("warning") {
        promises.push("Danger ahead for the protagonist".to_string());
    }
    if lower.contains("mystery") || lower.contains("strange") || lower.contains("unexplained") {
        promises.push("A mystery to solve".to_string());
    }
    if lower.contains("love") || lower.contains("attraction") || lower.contains("beautiful") {
        promises.push("A romantic element".to_string());
    }
    if lower.contains("quest") || lower.contains("journey") || lower.contains("mission") {
        promises.push("A journey or quest".to_string());
    }
    if lower.contains("revenge") || lower.contains("justice") || lower.contains("wrong") {
        promises.push("Justice or revenge to pursue".to_string());
    }
    if lower.contains("dead") || lower.contains("death") || lower.contains("kill") {
        promises.push("Death and its consequences".to_string());
    }
    if text.contains('?') {
        promises.push("A question to answer".to_string());
    }

    promises.dedup();
    promises
}

fn count_characters_introduced(text: &str) -> usize {
    // Heuristic: count capitalized words that appear in dialogue attribution or as subjects
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut names: Vec<String> = Vec::new();

    for (i, word) in words.iter().enumerate() {
        // Skip first word of sentences
        if i == 0 {
            continue;
        }
        let prev = if i > 0 { words[i - 1] } else { "" };
        let is_sentence_start = prev.ends_with('.') || prev.ends_with('!') || prev.ends_with('?');
        if is_sentence_start {
            continue;
        }

        if word
            .chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false)
            && word.len() > 1
            && word.chars().all(|c| c.is_alphabetic())
        {
            let name = word.to_string();
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }

    // Filter out common non-name capitalized words
    let common_words = [
        "The", "This", "That", "These", "Those", "His", "Her", "Its", "My", "Your", "Our", "Their",
        "He", "She", "It", "We", "They", "Mr", "Mrs", "Ms",
    ];
    names.retain(|n| !common_words.contains(&n.as_str()));

    names.len().min(10)
}

/// Returns per-paragraph engagement scores derived purely from structural signals:
/// dialogue presence, question marks, paragraph length, and backstory markers.
/// No IMAGE/ACTION wordlist hits are used here.
fn engagement_per_paragraph(paragraphs: &[&str]) -> Vec<f64> {
    paragraphs
        .iter()
        .map(|para| {
            let lower = para.to_lowercase();
            let word_count = para.split_whitespace().count();
            let mut score = 0.5;

            // Dialogue boosts engagement (structural: punctuation-based)
            if para.contains('"') || para.contains('\u{201C}') {
                score += 0.2;
            }

            // Questions boost (structural: punctuation-based)
            let question_count = para.chars().filter(|c| *c == '?').count();
            score += (question_count as f64 * 0.1).min(0.2);

            // Short paragraphs signal action/urgency (structural: word count)
            if word_count < 20 {
                score += 0.1;
            }

            // Very long paragraphs reduce engagement (structural: word count)
            if word_count > 200 {
                score -= 0.1;
            }

            // Backstory penalty (structural: tense/phrase markers)
            let backstory = BACKSTORY_MARKERS
                .iter()
                .filter(|m| lower.contains(*m))
                .count();
            score -= backstory as f64 * 0.1;

            score.clamp(0.1, 1.0)
        })
        .collect()
}

/// Computes the paragraph-length trajectory: word counts per paragraph.
fn paragraph_word_counts(paragraphs: &[&str]) -> Vec<usize> {
    paragraphs
        .iter()
        .map(|p| p.split_whitespace().count())
        .collect()
}

/// Returns the index of the first paragraph containing dialogue, or None.
fn dialogue_onset(paragraphs: &[&str]) -> Option<usize> {
    paragraphs
        .iter()
        .position(|p| p.contains('"') || p.contains('\u{201C}') || p.contains('\u{2018}'))
}

pub fn analyze_opening(
    text: &str,
    genre: &str,
    full_manuscript_voice_fingerprint: Option<&[f64]>,
) -> OpeningAnalysisResult {
    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();

    let first_line = text.lines().next().unwrap_or("").trim();
    let first_paragraph = paragraphs.first().copied().unwrap_or("");

    // Hook classification
    let (hook_type, hook_score) = classify_hook(first_line, first_paragraph);

    // --- Structural sub-signals ---

    // First-line word count
    let first_line_word_count = first_line.split_whitespace().count();

    // Paragraph-length trajectory
    let para_word_counts = paragraph_word_counts(&paragraphs);

    // Dialogue onset
    let dialogue_onset_paragraph = dialogue_onset(&paragraphs);

    // First line quality: purely structural (word count range, backstory penalty)
    let first_line_quality = {
        let mut q: f64 = 0.5;
        // Ideal length: 5–25 words
        if (5..=25).contains(&first_line_word_count) {
            q += 0.3;
        } else if first_line_word_count > 40 {
            q -= 0.2;
        }
        // Penalty for starting with backstory markers
        let lower = first_line.to_lowercase();
        if BACKSTORY_MARKERS.iter().any(|m| lower.contains(*m)) {
            q -= 0.3;
        }
        q.clamp(0.0, 1.0)
    };

    // First paragraph quality: structural (word count range, backstory ratio, dialogue)
    let first_paragraph_quality = {
        let mut q = 0.5;
        let para_words = first_paragraph.split_whitespace().count();
        if (20..=150).contains(&para_words) {
            q += 0.15;
        }
        let backstory_ratio_para = calculate_backstory_ratio(first_paragraph);
        q -= backstory_ratio_para * 0.3;
        if first_paragraph.contains('"') || first_paragraph.contains('\u{201C}') {
            q += 0.1;
        }
        q.clamp(0.0, 1.0)
    };

    // Backstory ratio
    let backstory_ratio = calculate_backstory_ratio(text);

    // Voice establishment (structural: first-person pronouns, long-word density)
    let voice_establishment = {
        let mut score: f64 = 0.5;
        let lower = first_paragraph.to_lowercase();
        if lower.starts_with("i ") || lower.contains(" i ") {
            score += 0.3;
        }
        let words: Vec<&str> = first_paragraph.split_whitespace().collect();
        let long_words = words.iter().filter(|w| w.len() > 8).count();
        if long_words > 2 {
            score += 0.1;
        }
        if let Some(fp) = full_manuscript_voice_fingerprint
            && !fp.is_empty()
        {
            score += 0.1;
        }
        score.clamp(0.0, 1.0)
    };

    // Engagement trajectory (structural signals only)
    let engagement_trajectory = engagement_per_paragraph(&paragraphs);

    // Promises
    let promises_made = detect_promises(text);

    // Characters introduced
    let characters_introduced = count_characters_introduced(text);

    // World establishment
    let lower_text = text.to_lowercase();
    let world_established = SITUATION_INDICATORS
        .iter()
        .filter(|s| lower_text.contains(*s))
        .count()
        >= 2
        || IMAGE_INDICATORS
            .iter()
            .filter(|s| lower_text.contains(*s))
            .count()
            >= 3;

    // Genre alignment
    let (preferred_hooks, genre_weight) = genre_opening_preferences(genre);
    let hook_type_name = match &hook_type {
        Some(HookType::Action) => "action",
        Some(HookType::Question) => "question",
        Some(HookType::Voice) => "voice",
        Some(HookType::Situation) => "situation",
        Some(HookType::Character) => "character",
        Some(HookType::Image) => "image",
        None => "",
    };
    let genre_alignment = if preferred_hooks.contains(&hook_type_name) {
        genre_weight
    } else {
        genre_weight * 0.5
    };

    // KISS: read_on_prediction is built only from structural sub-signals
    // (word counts, dialogue/backstory markers, paragraph variety); IMAGE/ACTION
    // wordlist hits are deliberately excluded as measured-unreliable.
    let read_on_prediction = {
        let avg_engagement = if engagement_trajectory.is_empty() {
            0.5
        } else {
            engagement_trajectory.iter().sum::<f64>() / engagement_trajectory.len() as f64
        };

        // Dialogue onset bonus: dialogue in first 2 paragraphs = +0.1
        let dialogue_bonus = match dialogue_onset_paragraph {
            Some(idx) if idx < 2 => 0.1,
            Some(_) => 0.05,
            None => 0.0,
        };

        // Paragraph-length variety: coefficient of variation of word counts
        let para_variety = if para_word_counts.len() >= 2 {
            let mean =
                para_word_counts.iter().sum::<usize>() as f64 / para_word_counts.len() as f64;
            if mean > 0.0 {
                let variance = para_word_counts
                    .iter()
                    .map(|&c| {
                        let d = c as f64 - mean;
                        d * d
                    })
                    .sum::<f64>()
                    / para_word_counts.len() as f64;
                let cv = variance.sqrt() / mean;
                // CV > 0.5 = good variety; cap contribution at 0.1
                (cv * 0.1).min(0.1)
            } else {
                0.0
            }
        } else {
            0.0
        };

        (first_line_quality * 0.25
            + (1.0 - backstory_ratio) * 0.20
            + voice_establishment * 0.15
            + avg_engagement * 0.15
            + dialogue_bonus
            + para_variety
            + genre_alignment * 0.15)
            .clamp(0.0, 1.0)
    };

    // Issues and strengths
    let mut issues = Vec::new();
    let mut strengths = Vec::new();

    if backstory_ratio > 0.4 {
        issues.push(
            "Opening is heavy with backstory/exposition; consider starting in media res"
                .to_string(),
        );
    }
    if first_line_word_count > 40 {
        issues.push(
            "First sentence is very long; consider a shorter, punchier opening line".to_string(),
        );
    }
    if characters_introduced > 4 {
        issues.push(format!(
            "Too many characters introduced ({}) in the opening; this can overwhelm readers",
            characters_introduced
        ));
    }
    if hook_type.is_none() {
        issues.push("No clear hook detected in the opening; the first page should create an immediate reason to read on".to_string());
    }
    if !world_established {
        issues.push("Setting not clearly established in opening pages".to_string());
    }
    if engagement_trajectory.len() > 2 {
        let last_few: Vec<f64> = engagement_trajectory
            .iter()
            .rev()
            .take(2)
            .copied()
            .collect();
        if last_few.iter().all(|&e| e < 0.4) {
            issues.push("Engagement drops in later opening paragraphs; consider adding a complication or question".to_string());
        }
    }
    if dialogue_onset_paragraph.is_none() && paragraphs.len() > 3 {
        issues.push(
            "No dialogue in the opening; consider introducing a voice early to anchor the reader"
                .to_string(),
        );
    }

    if first_line_quality > 0.7 {
        strengths.push("Excellent first line with strong specificity and energy".to_string());
    }
    if backstory_ratio < 0.1 {
        strengths.push("Opening is present-focused with minimal backstory".to_string());
    }
    if voice_establishment > 0.7 {
        strengths.push("Distinctive voice established immediately".to_string());
    }
    if !promises_made.is_empty() {
        strengths.push(format!(
            "Clear narrative promises made: {} implicit questions raised",
            promises_made.len()
        ));
    }
    if dialogue_onset_paragraph.map(|i| i < 2).unwrap_or(false) {
        strengths
            .push("Dialogue appears early, grounding the reader in character voice".to_string());
    }

    // Agent perspective — what a literary agent thinks after reading page 1
    let agent_perspective = if read_on_prediction > 0.75 {
        format!(
            "Strong opening. The {} hook creates immediate engagement, and the voice is distinctive enough to sustain interest. An agent would likely read to page 10.",
            hook_type_name
        )
    } else if hook_type.is_some() && backstory_ratio < 0.5 {
        format!(
            "Competent opening but not yet compelling. The writing is clean but the reader lacks a REASON to continue — {}. An agent might skim to page 5 looking for a reason to invest.",
            if promises_made.is_empty() { "no narrative question is raised" }
            else if backstory_ratio > 0.3 { "too much backstory delays the story's start" }
            else { "the stakes aren't clear enough yet" }
        )
    } else {
        format!(
            "This opening would likely result in a form rejection. The primary issue: {}. Agents read hundreds of openings weekly — yours needs to be undeniable within 3 paragraphs.",
            if backstory_ratio > 0.4 { "the story hasn't started — the reader is receiving history, not experiencing story" }
            else if hook_score < 0.3 { "there's no hook — nothing makes the reader need to know what happens next" }
            else if characters_introduced > 5 { "too many characters overwhelm before the reader has invested in any one" }
            else { "the writing lacks the specificity and energy that signals a distinctive voice" }
        )
    };

    // First line feedback
    let first_line_feedback = if first_line_quality > 0.7 {
        "Your opening line works. It's specific, energetic, and immediately places the reader in a moment.".to_string()
    } else if first_line_quality > 0.4 {
        "Your opening line is adequate but forgettable. A great first line is either: (1) an image so specific it could only exist in THIS story, (2) a voice so distinctive the reader trusts it immediately, or (3) a question so compelling the reader must have the answer.".to_string()
    } else {
        "Your opening line isn't doing enough work. It should be the most carefully crafted sentence in the entire manuscript. Every word must earn its place. Try: start with action, start with a contradiction, or start with the most interesting moment in the first chapter and work backward.".to_string()
    };

    // Revision suggestion
    let revision_suggestion = if backstory_ratio > 0.3 && hook_score < 0.5 {
        "Start later. Find the first moment of CONFLICT or DECISION in your opening chapter and begin there. Move backstory to chapter 2-3, delivered through character action, not narration.".to_string()
    } else if hook_score > 0.6 && voice_establishment < 0.4 {
        "Your hook is working but your voice isn't distinctive yet. Read your opening aloud — does it sound like only YOU could have written it? Infuse the narration with your protagonist's worldview, vocabulary, and rhythm.".to_string()
    } else if characters_introduced > 4 {
        format!("Reduce opening characters to 2-3. Introduce {} others later, after the reader has someone to care about.", characters_introduced - 2)
    } else if engagement_trajectory.len() > 3
        && engagement_trajectory.last().copied().unwrap_or(0.5) < 0.3
    {
        "Your opening starts well but loses momentum. End your first scene on a QUESTION — something the reader must turn the page to answer. Not a mystery; a human question about what your character will do next.".to_string()
    } else {
        "Consider: what would happen if you cut your first paragraph entirely? Often the real opening is hiding in paragraph 2 or 3.".to_string()
    };

    OpeningAnalysisResult {
        hook_type,
        hook_score,
        first_line_quality,
        first_paragraph_quality,
        backstory_ratio,
        voice_establishment,
        engagement_trajectory,
        promises_made,
        characters_introduced,
        world_established,
        genre_alignment,
        read_on_prediction,
        first_line_word_count,
        paragraph_word_counts: para_word_counts,
        dialogue_onset_paragraph,
        issues,
        strengths,
        agent_perspective,
        first_line_feedback,
        revision_suggestion,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_opening() {
        let text = "The bullet shattered the window and she dove behind the counter. Glass rained across the tile floor. Another shot fired. She sprinted for the back door.";
        let result = analyze_opening(text, "thriller", None);
        assert!(matches!(result.hook_type, Some(HookType::Action)));
        assert!(result.hook_score > 0.5);
    }

    #[test]
    fn test_backstory_heavy_opening() {
        let text = "She had always been a quiet child. Growing up in the small town, she had been known for her love of books. Years ago, her mother used to read to her every night. Back when things were simpler, the family had been happy.";
        let result = analyze_opening(text, "literary", None);
        assert!(result.backstory_ratio > 0.3);
        assert!(result.issues.iter().any(|i| i.contains("backstory")));
    }

    #[test]
    fn test_question_hook() {
        let text = "Who killed Marcus Webb? That was the question that had haunted Detective Sano for three months. The case file sat on her desk, growing thicker but no closer to an answer.";
        let result = analyze_opening(text, "mystery", None);
        assert!(matches!(result.hook_type, Some(HookType::Question)));
    }

    #[test]
    fn test_engagement_trajectory_non_empty() {
        let text = "First paragraph here.\n\nSecond paragraph with some action and running.\n\nThird paragraph with dialogue: \"Hello,\" she said.";
        let result = analyze_opening(text, "literary", None);
        assert_eq!(result.engagement_trajectory.len(), 3);
    }

    #[test]
    fn test_first_line_word_count_structural_signal() {
        let text = "She ran.\n\nThe rest of the story follows.";
        let result = analyze_opening(text, "thriller", None);
        // "She ran." = 2 words
        assert_eq!(result.first_line_word_count, 2);
    }

    #[test]
    fn test_paragraph_word_counts_trajectory() {
        let text = "Short line.\n\nThis is a somewhat longer paragraph with more words in it to test the trajectory signal.\n\nEnd.";
        let result = analyze_opening(text, "literary", None);
        assert_eq!(result.paragraph_word_counts.len(), 3);
        // Middle paragraph should be longest
        assert!(result.paragraph_word_counts[1] > result.paragraph_word_counts[0]);
        assert!(result.paragraph_word_counts[1] > result.paragraph_word_counts[2]);
    }

    #[test]
    fn test_dialogue_onset_detected() {
        let text = "The morning was cold.\n\n\"Get up,\" she said. \"We need to move.\"\n\nHe rolled over and stared at the ceiling.";
        let result = analyze_opening(text, "thriller", None);
        // Dialogue is in paragraph index 1
        assert_eq!(result.dialogue_onset_paragraph, Some(1));
    }

    #[test]
    fn test_dialogue_onset_none_when_absent() {
        let text =
            "The morning was cold. She walked to the window. Outside, the city was waking up.";
        let result = analyze_opening(text, "literary", None);
        assert_eq!(result.dialogue_onset_paragraph, None);
    }

    #[test]
    fn test_read_on_prediction_no_wordlist_inflation() {
        // A text with zero IMAGE/ACTION words but strong structural signals:
        // short first line, early dialogue, no backstory
        let text = "She stopped.\n\n\"Now,\" he said.\n\nThe door opened.";
        let result = analyze_opening(text, "thriller", None);
        // read_on_prediction should be > 0.5 due to structural signals alone
        assert!(
            result.read_on_prediction > 0.4,
            "read_on_prediction={} should be driven by structure",
            result.read_on_prediction
        );
        // Dialogue onset should be early
        assert_eq!(result.dialogue_onset_paragraph, Some(1));
    }

    #[test]
    fn test_backstory_heavy_lowers_read_on() {
        let text = "She had always been a quiet child. Growing up in the small town, she had been known for her love of books. Years ago, her mother used to read to her every night. Back when things were simpler, the family had been happy. It all started long before she was born, in the old days when the family had been different.";
        let result = analyze_opening(text, "literary", None);
        // Heavy backstory should suppress read_on_prediction
        assert!(
            result.read_on_prediction < 0.7,
            "read_on_prediction={} should be suppressed by backstory",
            result.read_on_prediction
        );
        assert!(result.backstory_ratio > 0.5);
    }
}
