use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{DialogueLine, speech_verbs};
use crate::{
    Result,
    craft::lexicon::is_word_char as word_char,
    manuscript::TextRange,
    substrate::{common_types::invalid, offsets::OffsetIndex, utils::AnalysisAvailability},
};

const MAX_TEXT_BYTES: usize = 16 * 1024 * 1024;
const MAX_LINES: usize = 10_000;
const MAX_DEPTH: usize = 32;
const TAG_WINDOW: usize = 512;
const NAME_WINDOW: usize = 256;

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DialogueAttributionMode {
    /// Explicit speech tags only, plus the narrative model's speaker when a
    /// verified model is configured.
    #[default]
    ExplicitOnly,
    /// Also proximity and alternation guesses for quotes neither a tag nor the
    /// model attributes.
    ConservativeHeuristic,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DialogueExtractionOptions {
    #[serde(default)]
    pub attribution: DialogueAttributionMode,
    #[serde(default)]
    pub include_indirect: bool,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DialogueKind {
    Direct,
    Indirect,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LocatedDialogueLine {
    pub line: DialogueLine,
    /// Selects the exact original line.quote; delimiters and outside whitespace are excluded.
    pub range: TextRange,
    pub quotation_range: Option<TextRange>,
    pub attribution_range: Option<TextRange>,
    pub kind: DialogueKind,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DialogueExtractionResult {
    pub availability: AnalysisAvailability,
    pub diagnostics: Vec<String>,
    pub lines: Vec<LocatedDialogueLine>,
    /// The verified narrative model that attributed speakers, when one did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_identity: Option<crate::onnx::model::ModelIdentity>,
}

fn name_char(ch: char) -> bool {
    word_char(ch) || matches!(ch, '\'' | '’' | '-')
}
fn normalize_name(text: &str) -> String {
    text.split_whitespace()
        .map(|part| part.replace('’', "'").to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Clone)]
struct NameIdentity {
    speaker: Option<String>,
    full: bool,
}

fn name_title(word: &str) -> bool {
    matches!(
        word.to_ascii_lowercase().as_str(),
        "mr" | "mrs" | "ms" | "dr" | "sir" | "lady" | "lord" | "miss"
    )
}

fn incompatible_before(text: &str) -> bool {
    let text = text.trim_end_matches([' ', '\t']);
    if !text.chars().next_back().is_some_and(word_char) {
        return false;
    }
    let start = text
        .char_indices()
        .rev()
        .find(|(_, ch)| !word_char(*ch))
        .map_or(0, |(i, ch)| i + ch.len_utf8());
    let word = &text[start..];
    let lower = word.to_ascii_lowercase();
    word.chars().next().is_some_and(char::is_uppercase)
        && !name_title(word)
        && !speech_verbs().contains(lower.as_str())
        && !matches!(
            lower.as_str(),
            "and" | "but" | "then" | "so" | "now" | "finally" | "suddenly" | "meanwhile"
        )
}

fn incompatible_after(text: &str, end: usize) -> bool {
    let remainder = text[end..].trim_start_matches([' ', '\t']);
    if !remainder.chars().next().is_some_and(char::is_uppercase) {
        return false;
    }
    !speech_verbs().contains(edge_word(remainder, false).as_str())
}

#[derive(Default)]
struct NameNode {
    children: BTreeMap<char, usize>,
    terminal: Option<NameIdentity>,
}
struct Names {
    forward: Vec<NameNode>,
    backward: Vec<NameNode>,
}
struct NameMatch {
    speaker: Option<String>,
    start: usize,
    end: usize,
}

impl Names {
    fn new(cast: &[String]) -> Self {
        let mut full = BTreeMap::<String, String>::new();
        for name in cast {
            full.entry(normalize_name(name))
                .and_modify(|value| {
                    if name < value {
                        *value = name.clone();
                    }
                })
                .or_insert_with(|| name.clone());
        }
        let mut aliases = BTreeMap::<String, BTreeSet<String>>::new();
        for (key, canonical) in &full {
            aliases
                .entry(key.clone())
                .or_default()
                .insert(canonical.clone());
            let parts: Vec<_> = key.split_whitespace().collect();
            for short in [parts.first().copied(), parts.last().copied()]
                .into_iter()
                .flatten()
            {
                if !full.contains_key(short)
                    && ![
                        "a", "an", "the", "mr.", "mrs.", "ms.", "dr.", "sir", "lady", "lord",
                    ]
                    .contains(&short)
                {
                    aliases
                        .entry(short.into())
                        .or_default()
                        .insert(canonical.clone());
                }
            }
        }
        let mut names = Self {
            forward: vec![NameNode::default()],
            backward: vec![NameNode::default()],
        };
        for (alias, identities) in aliases {
            let terminal = NameIdentity {
                full: full.contains_key(&alias),
                speaker: if identities.len() == 1 {
                    identities.into_iter().next()
                } else {
                    None
                },
            };
            Self::insert(&mut names.forward, alias.chars(), terminal.clone());
            Self::insert(&mut names.backward, alias.chars().rev(), terminal);
        }
        names
    }
    fn insert(
        nodes: &mut Vec<NameNode>,
        chars: impl Iterator<Item = char>,
        terminal: NameIdentity,
    ) {
        let mut node = 0;
        for ch in chars {
            node = if let Some(&child) = nodes[node].children.get(&ch) {
                child
            } else {
                let child = nodes.len();
                nodes.push(NameNode::default());
                nodes[node].children.insert(ch, child);
                child
            };
        }
        nodes[node].terminal = Some(terminal);
    }
    fn prefix(&self, text: &str) -> Option<NameMatch> {
        let trimmed = text.trim_start();
        let base = text.len() - trimmed.len();
        let mut node = 0;
        let mut space = false;
        let mut best = None;
        for (position, ch) in trimmed.char_indices() {
            if position > NAME_WINDOW {
                break;
            }
            let mapped = if ch.is_whitespace() {
                if space {
                    continue;
                }
                space = true;
                ' '
            } else {
                space = false;
                if ch == '’' { '\'' } else { ch }
            };
            for unit in mapped.to_lowercase() {
                let Some(&next) = self.forward[node].children.get(&unit) else {
                    return best;
                };
                node = next;
            }
            let end = base + position + ch.len_utf8();
            if text[end..]
                .chars()
                .next()
                .is_none_or(|next| !name_char(next))
                && let Some(identity) = &self.forward[node].terminal
            {
                best = Some(NameMatch {
                    speaker: if !identity.full && incompatible_after(text, end) {
                        None
                    } else {
                        identity.speaker.clone()
                    },
                    start: base,
                    end,
                });
            }
        }
        best
    }
    fn suffix(&self, text: &str) -> Option<NameMatch> {
        let trimmed = text.trim_end();
        let mut node = 0;
        let mut space = false;
        let mut best = None;
        for (position, ch) in trimmed.char_indices().rev() {
            if trimmed.len() - position > NAME_WINDOW {
                break;
            }
            let mapped = if ch.is_whitespace() {
                if space {
                    continue;
                }
                space = true;
                ' '
            } else {
                space = false;
                if ch == '’' { '\'' } else { ch }
            };
            for unit in mapped.to_lowercase().rev() {
                let Some(&next) = self.backward[node].children.get(&unit) else {
                    return best;
                };
                node = next;
            }
            if trimmed[..position]
                .chars()
                .next_back()
                .is_none_or(|previous| !name_char(previous))
                && let Some(identity) = &self.backward[node].terminal
            {
                best = Some(NameMatch {
                    speaker: if !identity.full && incompatible_before(&trimmed[..position]) {
                        None
                    } else {
                        identity.speaker.clone()
                    },
                    start: position,
                    end: trimmed.len(),
                });
            }
        }
        best
    }
    fn in_text(&self, text: &str) -> BTreeSet<String> {
        let mut found = BTreeSet::new();
        for (position, ch) in text.char_indices() {
            if word_char(ch)
                && text[..position]
                    .chars()
                    .next_back()
                    .is_none_or(|previous| !name_char(previous))
                && let Some(name) = self.prefix(&text[position..])
                && let Some(identity) = name.speaker
            {
                found.insert(identity);
            }
        }
        found
    }
}

#[derive(Clone, Copy)]
struct Quote {
    start: usize,
    end: usize,
    content_start: usize,
    content_end: usize,
}
pub(crate) struct QuotationScopes {
    pub spans: Vec<(usize, usize)>,
    pub diagnostics: Vec<String>,
    pub boundaries: Vec<usize>,
    quotes: Vec<Quote>,
}

struct ScannedQuotes {
    quotes: Vec<Quote>,
    diagnostics: Vec<String>,
    unclosed: Option<(usize, usize)>,
    boundaries: Vec<usize>,
}

struct Frame {
    opening: char,
    closing: char,
    start: usize,
    content_start: usize,
}

fn edge_word(text: &str, from_end: bool) -> String {
    if from_end {
        text.chars()
            .rev()
            .take_while(|ch| word_char(*ch))
            .take(16)
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>()
            .to_ascii_lowercase()
    } else {
        text.chars()
            .take_while(|ch| word_char(*ch))
            .take(16)
            .collect::<String>()
            .to_ascii_lowercase()
    }
}

fn lexical_apostrophe(
    text: &str,
    position: usize,
    ch: char,
    previous: Option<char>,
    next: Option<char>,
) -> bool {
    if !matches!(ch, '\'' | '‘' | '’') {
        return false;
    }
    if previous.is_some_and(word_char) && next.is_some_and(word_char) {
        return true;
    }
    if ch == '‘' {
        return false;
    }
    let after = &text[position + ch.len_utf8()..];
    let right = edge_word(after.trim_start_matches([' ', '\t']), false);
    if next.is_some_and(word_char)
        && matches!(
            right.as_str(),
            "tis" | "twas" | "twere" | "twill" | "em" | "cause" | "bout" | "round"
        )
    {
        return true;
    }
    let before = text[..position].trim_end_matches([' ', '\t']);
    if !before.chars().next_back().is_some_and(word_char) {
        return false;
    }
    let left = edge_word(before, true);
    let pronoun = matches!(
        left.as_str(),
        "i" | "you"
            | "he"
            | "she"
            | "it"
            | "we"
            | "they"
            | "who"
            | "what"
            | "where"
            | "when"
            | "how"
            | "why"
            | "there"
            | "that"
    );
    match right.as_str() {
        "s" => true,
        "m" => left == "i",
        "d" | "ll" => pronoun,
        "re" => matches!(
            left.as_str(),
            "you" | "we" | "they" | "who" | "what" | "where" | "how" | "there"
        ),
        "ve" => {
            pronoun
                || matches!(
                    left.as_str(),
                    "would" | "could" | "should" | "must" | "might"
                )
        }
        "t" => left.ends_with('n'),
        _ => false,
    }
}

fn scan_quotes(text: &str) -> Result<ScannedQuotes> {
    let mut completed = Vec::<Quote>::new();
    let mut boundaries = Vec::new();
    let mut stack = Vec::<Frame>::new();
    let mut chars = text.char_indices().peekable();
    let mut previous = None;
    let mut slashes = 0;
    let mut line_start = true;
    let mut line_breaks = 0usize;
    let mut malformed = false;
    let mut root_count = 0usize;
    while let Some((position, ch)) = chars.next() {
        let next = chars.peek().map(|(_, ch)| *ch);
        let apostrophe = lexical_apostrophe(text, position, ch, previous, next);
        let escaped = slashes % 2 == 1;
        let continuation = stack.len() == 1
            && stack[0].opening == ch
            && line_start
            && (matches!(ch, '“' | '‘') || line_breaks >= 2)
            && text[position + ch.len_utf8()..]
                .chars()
                .find(|c| !c.is_whitespace())
                .is_some_and(word_char);
        if !escaped && !apostrophe && continuation {
            boundaries.push(position);
        }
        if !escaped
            && !apostrophe
            && !continuation
            && matches!(ch, '“' | '”' | '‘' | '’' | '"' | '\'')
        {
            if let Some(index) = stack.iter().rposition(|frame| frame.closing == ch) {
                boundaries.push(position);
                malformed |= index + 1 != stack.len();
                stack.truncate(index + 1);
                let frame = stack.pop().expect("closing frame exists");
                let content = &text[frame.content_start..position];
                let trimmed = content.trim();
                if !trimmed.is_empty() {
                    let start = frame.content_start + content.len() - content.trim_start().len();
                    completed.push(Quote {
                        start: frame.start,
                        end: position + ch.len_utf8(),
                        content_start: start,
                        content_end: start + trimmed.len(),
                    });
                    if stack.is_empty() {
                        root_count += 1;
                    }
                    if root_count > MAX_LINES || completed.len() > MAX_LINES * MAX_DEPTH {
                        return Err(invalid(
                            "dialogue quote count exceeds bounded extraction budget",
                        ));
                    }
                }
            } else {
                let closing = match ch {
                    '“' => Some('”'),
                    '‘' => Some('’'),
                    '"' => Some('"'),
                    '\'' => Some('\''),
                    _ => None,
                };
                if let Some(closing) = closing {
                    if matches!(ch, '\'' | '"') && previous.is_some_and(word_char) {
                    } else {
                        if stack.len() == MAX_DEPTH {
                            return Err(invalid("dialogue quote nesting exceeds 32"));
                        }
                        boundaries.push(position);
                        stack.push(Frame {
                            opening: ch,
                            closing,
                            start: position,
                            content_start: position + ch.len_utf8(),
                        });
                    }
                } else if matches!(ch, '”' | '’') && !previous.is_some_and(word_char) {
                    boundaries.push(position);
                    malformed = true;
                }
            }
        }
        if boundaries.len() > 2 * MAX_LINES * MAX_DEPTH + MAX_DEPTH {
            return Err(invalid(
                "dialogue delimiter count exceeds bounded extraction budget",
            ));
        }
        if matches!(ch, '\n' | '\r') {
            line_start = true;
            if ch != '\n' || previous != Some('\r') {
                line_breaks = (line_breaks + 1).min(2);
            }
        } else if !ch.is_whitespace() {
            line_start = false;
            line_breaks = 0;
        }
        slashes = if ch == '\\' { slashes + 1 } else { 0 };
        previous = Some(ch);
    }
    completed.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| b.end.cmp(&a.end)));
    let mut quotes = Vec::<Quote>::new();
    for quote in completed {
        if quotes
            .last()
            .is_none_or(|previous| quote.start >= previous.end)
        {
            quotes.push(quote);
        }
        if quotes.len() > MAX_LINES {
            return Err(invalid("dialogue line count exceeds 10000"));
        }
    }
    let diagnostics = if malformed || !stack.is_empty() {
        vec!["unbalanced_quote_delimiters".into()]
    } else {
        Vec::new()
    };
    Ok(ScannedQuotes {
        quotes,
        diagnostics,
        unclosed: stack.first().map(|frame| (frame.start, text.len())),
        boundaries,
    })
}

/// Maximal source scopes, including unmatched opener-to-EOF barriers. Balanced
/// descendants recovered as dialogue remain inside an unclosed parent scope.
pub(crate) fn quotation_scopes(text: &str) -> Result<super::QuotationScopes> {
    if text.len() > MAX_TEXT_BYTES {
        return Err(invalid("quotation source exceeds 16 MiB"));
    }
    let scanned = scan_quotes(text)?;
    let mut spans: Vec<_> = scanned
        .quotes
        .iter()
        .map(|quote| (quote.start, quote.end))
        .chain(scanned.unclosed)
        .collect();
    spans.sort_unstable();
    let mut merged = Vec::<(usize, usize)>::new();
    for (start, end) in spans {
        if let Some(previous) = merged.last_mut()
            && start <= previous.1
        {
            previous.1 = previous.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    let result = QuotationScopes {
        spans: merged,
        diagnostics: scanned.diagnostics,
        boundaries: scanned.boundaries,
        quotes: scanned.quotes,
    };
    debug_assert!(result.boundaries.windows(2).all(|pair| pair[0] < pair[1]));
    Ok(result)
}

fn words(text: &str) -> Vec<(usize, usize)> {
    let mut found = Vec::new();
    let mut start = None;
    for (position, ch) in text.char_indices() {
        if word_char(ch) {
            start.get_or_insert(position);
        } else if let Some(begin) = start.take() {
            found.push((begin, position));
        }
    }
    if let Some(begin) = start {
        found.push((begin, text.len()));
    }
    found
}
fn adverb(word: &str) -> bool {
    word.chars().next().is_some_and(char::is_lowercase) && word.ends_with("ly")
}

fn verb_at(text: &str, names: &Names) -> Option<(usize, usize, String)> {
    let trimmed = text.trim_start();
    let base = text.len() - trimmed.len();
    let end = trimmed
        .char_indices()
        .find(|(_, ch)| !word_char(*ch))
        .map_or(trimmed.len(), |(i, _)| i);
    if end == 0 {
        return None;
    }
    let word = trimmed[..end].to_lowercase();
    if speech_verbs().contains(word.as_str()) {
        Some((base, base + end, word))
    } else if adverb(&trimmed[..end]) && names.prefix(trimmed).is_none() {
        verb_at_no_adverb(&text[base + end..]).map(|(a, b, v)| (base + end + a, base + end + b, v))
    } else {
        None
    }
}
fn verb_at_no_adverb(text: &str) -> Option<(usize, usize, String)> {
    let trimmed = text.trim_start();
    let base = text.len() - trimmed.len();
    let end = trimmed
        .char_indices()
        .find(|(_, ch)| !word_char(*ch))
        .map_or(trimmed.len(), |(i, _)| i);
    let word = trimmed[..end].to_lowercase();
    speech_verbs()
        .contains(word.as_str())
        .then_some((base, base + end, word))
}

struct Tag {
    speaker: Option<String>,
    verb: String,
    start: usize,
    end: usize,
}
fn after_tag(text: &str, names: &Names) -> Option<Tag> {
    let clean = text.trim_start_matches(|ch: char| ch.is_whitespace() || matches!(ch, ',' | ':'));
    let base = text.len() - clean.len();
    if let Some(name) = names.prefix(clean)
        && let Some((_, end, verb)) = verb_at(&clean[name.end..], names)
    {
        return Some(Tag {
            speaker: name.speaker,
            verb,
            start: base + name.start,
            end: base + name.end + end,
        });
    }
    if let Some((start, end, verb)) = verb_at_no_adverb(clean)
        && start == 0
    {
        let mut remainder = &clean[end..];
        let mut offset = end;
        let immediate = names.prefix(remainder);
        if immediate.is_none()
            && let Some((a, b)) = words(remainder).first().copied()
            && adverb(&remainder[a..b])
        {
            offset += b;
            remainder = &clean[offset..];
        }
        if let Some(name) = immediate.or_else(|| names.prefix(remainder)) {
            return Some(Tag {
                speaker: name.speaker,
                verb,
                start: base,
                end: base + offset + name.end,
            });
        }
        return Some(Tag {
            speaker: None,
            verb,
            start: base,
            end: base + end,
        });
    }
    for (a, b) in words(clean).into_iter().take(8) {
        let verb = clean[a..b].to_lowercase();
        if speech_verbs().contains(verb.as_str()) {
            return Some(Tag {
                speaker: None,
                verb,
                start: base,
                end: base + b,
            });
        }
        if clean[..a].contains(['.', ';', '!', '?', '\n', '\r']) {
            break;
        }
    }
    None
}
fn before_tag(text: &str, names: &Names) -> Option<Tag> {
    let clean = text.trim_end_matches(|ch: char| {
        ch.is_whitespace() || matches!(ch, ',' | ':' | '.' | '!' | '?' | ';')
    });
    if let Some(name) = names.suffix(clean) {
        let prefix = clean[..name.start].trim_end();
        if let Some((a, b)) = words(prefix).last().copied() {
            let verb = prefix[a..b].to_lowercase();
            if speech_verbs().contains(verb.as_str()) {
                return Some(Tag {
                    speaker: name.speaker,
                    verb,
                    start: a,
                    end: name.end,
                });
            }
        }
    }
    for (a, b) in words(clean).into_iter().rev() {
        let verb = clean[a..b].to_lowercase();
        if !speech_verbs().contains(verb.as_str()) {
            continue;
        }
        let tail = clean[b..].trim();
        let tail_words: Vec<_> = tail.split_whitespace().collect();
        let allowed = tail.is_empty()
            || tail_words.len() == 1 && adverb(tail_words[0])
            || tail_words
                .first()
                .is_some_and(|word| word.eq_ignore_ascii_case("to"))
                && tail_words.len() >= 2
                && !tail.contains(['.', ';', '!', '?'])
            || tail.eq_ignore_ascii_case("that");
        if !allowed {
            continue;
        }
        let mut subject = &clean[..a];
        let immediate = names.suffix(subject);
        if immediate.is_none()
            && let Some((adverb_start, adverb_end)) = words(subject).last().copied()
            && adverb(&subject[adverb_start..adverb_end])
        {
            subject = &subject[..adverb_start];
        }
        if let Some(name) = immediate.or_else(|| names.suffix(subject)) {
            return Some(Tag {
                speaker: name.speaker,
                verb,
                start: name.start,
                end: b,
            });
        }
        if !subject.trim().is_empty() {
            return Some(Tag {
                speaker: None,
                verb,
                start: 0,
                end: b,
            });
        }
    }
    None
}

fn continuation_tail(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.contains(',') && trimmed.chars().all(|ch| ch == ',' || ch.is_whitespace()) {
        return true;
    }
    let Some(recipient) = trimmed
        .strip_suffix(',')
        .and_then(|value| value.trim().strip_prefix("to "))
    else {
        return false;
    };
    !recipient.contains([',', '.', ';', '!', '?'])
        && (1..=4).contains(&recipient.split_whitespace().count())
}

fn lower_boundary(text: &str, mut position: usize) -> usize {
    while !text.is_char_boundary(position) {
        position += 1;
    }
    position
}
fn upper_boundary(text: &str, mut position: usize) -> usize {
    while !text.is_char_boundary(position) {
        position -= 1;
    }
    position
}
fn range(index: &OffsetIndex<'_>, start: usize, end: usize) -> TextRange {
    TextRange {
        byte_start: start,
        byte_end: end,
        utf16_start: index.utf16(start).expect("original boundary"),
        utf16_end: index.utf16(end).expect("original boundary"),
    }
}

/// Extract paired quotation candidates and explicit grammatical speakers. Rule
/// weights in legacy confidence are not calibrated attribution probabilities.
/// Unquoted speech and speech-versus-title semantics are not inferred.
pub fn extract_dialogue_checked(
    text: &str,
    scene_id: usize,
    cast: &[String],
    options: &DialogueExtractionOptions,
) -> Result<DialogueExtractionResult> {
    if text.len() > MAX_TEXT_BYTES || cast.len() > 256 {
        return Err(invalid("dialogue exceeds 16 MiB or 256 characters"));
    }
    for name in cast {
        crate::manuscript::config::validate_name(name)?;
    }
    let names = Names::new(cast);
    let QuotationScopes {
        quotes,
        mut diagnostics,
        spans: scopes,
        boundaries: _,
    } = super::quotation_scopes(text)?;
    let mut offsets = if quotes.is_empty() {
        None
    } else {
        Some(OffsetIndex::new(text)?)
    };
    // The narrative model scores a speaker for every quotation at once; explicit
    // tags still win because they are exact, and the heuristics only fill quotes
    // the model leaves unattributed.
    let model = if quotes.is_empty() {
        None
    } else {
        let spans: Vec<_> = quotes
            .iter()
            .map(|quote| (quote.start, quote.end))
            .collect();
        match crate::continuity::narrative::analyze(text, &spans) {
            Ok(analysis) => Some(analysis),
            Err(failure) => {
                if failure.kind != crate::onnx::checked::FailureKind::Unavailable {
                    diagnostics.push(format!("model_fallback:{}", failure.code));
                }
                None
            }
        }
    };
    let normalize = |name: &str| {
        name.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    };
    let cast_names: std::collections::HashMap<String, &str> = cast
        .iter()
        .map(|name| (normalize(name), name.as_str()))
        .collect();
    let mut lines = Vec::new();
    let mut recent = Vec::<String>::new();
    let mut previous_speaker = None;
    let mut previous_after = None::<(usize, usize)>;
    for (position, quote) in quotes.iter().enumerate() {
        let index = offsets.as_ref().expect("quoted source index");
        let lo =
            lower_boundary(text, quote.start.saturating_sub(TAG_WINDOW)).max(if position == 0 {
                0
            } else {
                quotes[position - 1].end
            });
        let hi = upper_boundary(text, (quote.end + TAG_WINDOW).min(text.len()))
            .min(quotes.get(position + 1).map_or(text.len(), |q| q.start));
        let before = &text[lo..quote.start];
        let after = &text[quote.end..hi];
        let mut pre = before_tag(before, &names);
        if let Some(tag) = &mut pre {
            tag.start += lo;
            tag.end += lo;
        }
        let mut post = after_tag(after, &names);
        if let Some(tag) = &mut post {
            tag.start += quote.end;
            tag.end += quote.end;
        }
        let prior_after = previous_after;
        if pre.as_ref().is_some_and(|tag| {
            prior_after == Some((tag.start, tag.end))
                && !continuation_tail(&text[tag.end..quote.start])
        }) {
            pre = None;
        }
        previous_after = post.as_ref().map(|tag| (tag.start, tag.end));
        let unresolved = pre.as_ref().is_some_and(|tag| tag.speaker.is_none())
            || post.as_ref().is_some_and(|tag| tag.speaker.is_none());
        let conflict = matches!((&pre,&post), (Some(a),Some(b)) if a.speaker.is_some() && b.speaker.is_some() && a.speaker != b.speaker);
        let chosen = if unresolved || conflict {
            None
        } else {
            post.as_ref().or(pre.as_ref())
        };
        let mut speaker = "Unknown".to_string();
        let mut confidence = 0.0;
        let mut method = "unknown".to_string();
        let mut tag_verb = None;
        let mut attribution = None;
        if let Some(tag) = chosen {
            speaker = tag.speaker.clone().expect("resolved tag");
            confidence = if post.is_some() { 0.95 } else { 0.9 };
            method = "explicit_tag".into();
            tag_verb = Some(tag.verb.clone());
            attribution = Some(range(index, tag.start, tag.end));
        } else if let Some((name, probability)) = model.as_ref().and_then(|analysis| {
            analysis
                .speaker_name(position)
                .map(|name| (name, analysis.quotes[position].confidence))
        }) {
            speaker = cast_names
                .get(&normalize(name))
                .map_or_else(|| name.to_string(), |cast| (*cast).to_string());
            confidence = probability;
            method = "narrative_model".into();
        } else if options.attribution == DialogueAttributionMode::ConservativeHeuristic
            && !unresolved
            && !conflict
        {
            let proximity_before = if prior_after.is_some() {
                before
                    .rfind(['.', '!', '?', '\n', '\r'])
                    .map_or("", |position| &before[position + 1..])
            } else {
                before
            };
            let mut candidates = names.in_text(proximity_before);
            candidates.extend(names.in_text(after));
            if candidates.len() == 1 {
                speaker = candidates.into_iter().next().expect("one candidate");
                confidence = 0.65;
                method = "proximity".into();
            } else if candidates.is_empty()
                && recent.len() == 2
                && let Some(previous) = &previous_speaker
                && let Some(other) = recent.iter().find(|name| *name != previous)
            {
                speaker = other.clone();
                confidence = 0.45;
                method = "pattern".into();
            }
        }
        if speaker != "Unknown" {
            previous_speaker = Some(speaker.clone());
            if !recent.contains(&speaker) {
                recent.push(speaker.clone());
                if recent.len() > 2 {
                    recent.remove(0);
                }
            }
        }
        lines.push(LocatedDialogueLine {
            line: DialogueLine {
                scene_id,
                quote: text[quote.content_start..quote.content_end].into(),
                speaker,
                confidence,
                method,
                tag_verb,
            },
            range: range(index, quote.content_start, quote.content_end),
            quotation_range: Some(range(index, quote.start, quote.end)),
            attribution_range: attribution,
            kind: DialogueKind::Direct,
        });
        if conflict {
            diagnostics.push("conflicting_adjacent_speech_tags".into());
        }
        if unresolved {
            diagnostics.push("unresolved_adjacent_speech_tag".into());
        }
    }
    if options.include_indirect {
        let mut quote_position = 0;
        for (begin, end) in crate::substrate::utils::sentence_spans(text) {
            let sentence = &text[begin..end];
            for (a, ch) in sentence.char_indices() {
                if !word_char(ch) || sentence[..a].chars().next_back().is_some_and(name_char) {
                    continue;
                }
                let start = begin + a;
                while scopes
                    .get(quote_position)
                    .is_some_and(|scope| scope.1 <= start)
                {
                    quote_position += 1;
                }
                if scopes
                    .get(quote_position)
                    .is_some_and(|scope| scope.0 <= start)
                {
                    continue;
                }
                let Some(actor) = names.prefix(&text[start..end]) else {
                    continue;
                };
                let Some(speaker) = actor.speaker else {
                    continue;
                };
                let rest = &text[start + actor.end..end];
                let tokens = words(rest);
                let Some(&(vstart, vend)) = tokens.first() else {
                    continue;
                };
                if !rest[..vstart].trim().is_empty() {
                    continue;
                }
                let verb = rest[vstart..vend].to_lowercase();
                if ![
                    "told",
                    "informed",
                    "explained",
                    "mentioned",
                    "reminded",
                    "assured",
                    "convinced",
                    "warned",
                    "said",
                    "whispered",
                ]
                .contains(&verb.as_str())
                {
                    continue;
                }
                let complement = &rest[vend..];
                let lowered = complement.to_lowercase();
                if ![" that ", " about "]
                    .iter()
                    .any(|marker| format!(" {} ", lowered.trim()).contains(marker))
                {
                    continue;
                }
                let content_end = scopes
                    .get(quote_position)
                    .filter(|scope| scope.0 > start && scope.0 < end)
                    .map_or(end, |scope| scope.0);
                let original = text[start..content_end].trim_end();
                if original.len() <= actor.end + vend {
                    continue;
                }
                let stop = start + original.len();
                if offsets.is_none() {
                    offsets = Some(OffsetIndex::new(text)?);
                }
                let index = offsets.as_ref().expect("reported source index");
                lines.push(LocatedDialogueLine {
                    line: DialogueLine {
                        scene_id,
                        quote: text[start..stop].into(),
                        speaker,
                        confidence: 0.8,
                        method: "indirect_speech".into(),
                        tag_verb: Some(verb),
                    },
                    range: range(index, start, stop),
                    quotation_range: None,
                    attribution_range: Some(range(index, start, start + actor.end + vend)),
                    kind: DialogueKind::Indirect,
                });
                if lines.len() > MAX_LINES {
                    return Err(invalid("dialogue line count exceeds 10000"));
                }
                break;
            }
        }
    }
    lines.sort_by_key(|line| (line.range.byte_start, line.range.byte_end));
    if lines.iter().any(|line| line.line.speaker == "Unknown") {
        diagnostics.push("speaker_attribution_incomplete".into());
    }
    if lines.iter().any(|line| {
        matches!(
            line.line.method.as_str(),
            "proximity" | "pattern" | "continuation"
        )
    }) {
        diagnostics.push("heuristic_speakers_unverified".into());
    }
    let partial = !diagnostics.is_empty();
    diagnostics.push("quotation_candidates_are_not_verified_spoken_events".into());
    diagnostics.push("confidence_values_are_uncalibrated_rule_weights".into());
    diagnostics.sort();
    diagnostics.dedup();
    Ok(DialogueExtractionResult {
        availability: if partial {
            AnalysisAvailability::Partial
        } else {
            AnalysisAvailability::Complete
        },
        diagnostics,
        lines,
        model_identity: model.map(|analysis| analysis.identity),
    })
}

#[cfg(test)]
mod scope_tests {
    use super::*;
    #[test]
    fn shared_scopes_preserve_nested_original_ranges_and_spaced_apostrophes() {
        let text = "🜂 'Tis late. “They ’ll bring 'em and ‘nested’ words,” Alice said.";
        let scopes = super::super::quotation_scopes(text).unwrap();
        let start = text.find('“').unwrap();
        let end = text.find('”').unwrap() + '”'.len_utf8();
        assert_eq!(scopes.spans, [(start, end)]);
        assert!(scopes.diagnostics.is_empty());
        let extracted =
            extract_dialogue_checked(text, 0, &["Alice".into()], &Default::default()).unwrap();
        let range = extracted.lines[0].quotation_range.as_ref().unwrap();
        assert_eq!(scopes.spans, [(range.byte_start, range.byte_end)]);
    }
    #[test]
    fn unclosed_outer_scope_remains_a_barrier_despite_balanced_descendant_recovery() {
        let text = "Before ‘ unclosed parent with \"balanced\" inner words.";
        let scopes = super::super::quotation_scopes(text).unwrap();
        assert_eq!(scopes.spans, [(text.find('‘').unwrap(), text.len())]);
        assert!(
            scopes
                .diagnostics
                .contains(&"unbalanced_quote_delimiters".into())
        );
        let extracted = extract_dialogue_checked(text, 0, &[], &Default::default()).unwrap();
        assert_eq!(extracted.lines[0].line.quote, "balanced");
    }
    #[test]
    fn indirect_reports_inside_unclosed_quotation_are_not_promoted_to_facts() {
        let result = extract_dialogue_checked(
            "“Alice told Bob that the road was closed.",
            0,
            &["Alice".into(), "Bob".into()],
            &DialogueExtractionOptions {
                include_indirect: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.lines.is_empty());
        assert!(
            result
                .diagnostics
                .contains(&"unbalanced_quote_delimiters".into())
        );
    }
    #[test]
    fn nested_scope_boundaries_exclude_lexical_and_escaped_apostrophes() {
        let text = r#"🜂 “Alice ’s words: ‘she waits’ and \"not a delimiter\".”"#;
        let scopes = super::super::quotation_scopes(text).unwrap();
        let expected: Vec<_> = text
            .char_indices()
            .filter(|(_, ch)| matches!(ch, '“' | '”' | '‘' | '’'))
            .filter(|(index, _)| *index != text.find('’').unwrap())
            .map(|(i, _)| i)
            .collect();
        assert_eq!(scopes.boundaries, expected);
    }
    #[test]
    fn repeated_structural_delimiters_have_a_separate_bounded_metadata_budget() {
        let max = 2 * MAX_LINES * MAX_DEPTH + MAX_DEPTH;
        for count in [max - 1, max, max + 1] {
            let text = "“x\n\n".repeat(count);
            let result = super::super::quotation_scopes(&text);
            assert_eq!(result.is_ok(), count <= max);
            if let Ok(result) = result {
                assert_eq!(result.boundaries.len(), count);
            }
        }
    }
    #[test]
    fn shared_scope_input_byte_boundaries_are_checked() {
        for bytes in [MAX_TEXT_BYTES - 1, MAX_TEXT_BYTES, MAX_TEXT_BYTES + 1] {
            assert_eq!(
                super::super::quotation_scopes(&" ".repeat(bytes)).is_ok(),
                bytes <= MAX_TEXT_BYTES
            );
        }
    }
}
