use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::OnceLock;

use super::author_knowledge::AuthorSource;
use crate::{
    Error, Result,
    craft::{dialogue::quotation_scopes, lexicon::WordMatcher},
    manuscript::{
        Evidence, Manuscript, TextRange,
        author_source::validate_sources,
        config::{validate_id, validate_name},
    },
    substrate::{offsets::OffsetIndex, utils::AnalysisAvailability},
};

static MALE_PRONOUNS: &[&str] = &["he", "him", "his", "himself"];
static FEMALE_PRONOUNS: &[&str] = &["she", "her", "hers", "herself"];
static NEUTRAL_PRONOUNS: &[&str] = &["they", "them", "their", "theirs", "themselves"];
pub const MAX_COREFERENCE_MENTIONS: usize = 10_000;
pub const MAX_COREFERENCE_CHAIN_DEPTH: u8 = 16;
pub const MAX_AUTHOR_COREFERENCE_BYTES: usize = 4 * 1024 * 1024;
// Byte distances from a pronoun back to its candidate antecedent name.
const NEAR_ANTECEDENT_BYTES: usize = 80;
const FAR_ANTECEDENT_BYTES: usize = 200;

fn pronoun_matcher() -> &'static WordMatcher {
    static M: OnceLock<WordMatcher> = OnceLock::new();
    M.get_or_init(|| WordMatcher::new(&[MALE_PRONOUNS, FEMALE_PRONOUNS, NEUTRAL_PRONOUNS].concat()))
}
fn paragraph_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\n\s*\n").expect("constant paragraph regex"))
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Gender {
    Male,
    Female,
    Neutral,
    Unknown,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum PronounKind {
    Other,
    Possessive,
    Reflexive,
}
fn pronoun_gender(token: &str) -> Gender {
    let token = token.to_ascii_lowercase();
    if MALE_PRONOUNS.contains(&token.as_str()) {
        Gender::Male
    } else if FEMALE_PRONOUNS.contains(&token.as_str()) {
        Gender::Female
    } else {
        Gender::Neutral
    }
}
fn pronoun_kind(token: &str) -> PronounKind {
    match token.to_ascii_lowercase().as_str() {
        "himself" | "herself" | "themselves" => PronounKind::Reflexive,
        "his" | "her" | "their" => PronounKind::Possessive,
        _ => PronounKind::Other,
    }
}
fn gender_from_str(value: &str) -> Gender {
    match value {
        "male" => Gender::Male,
        "female" => Gender::Female,
        "neutral" => Gender::Neutral,
        _ => Gender::Unknown,
    }
}
fn compute_confidence(
    kind: PronounKind,
    exact_gender: bool,
    same_paragraph: bool,
    distance: usize,
    is_cataphoric: bool,
) -> f64 {
    if is_cataphoric {
        return 0.30;
    }
    match kind {
        PronounKind::Possessive | PronounKind::Reflexive => 0.40,
        _ if exact_gender && same_paragraph && distance < NEAR_ANTECEDENT_BYTES => 0.85,
        _ if exact_gender && distance < FAR_ANTECEDENT_BYTES => 0.70,
        _ if !exact_gender && same_paragraph => 0.55,
        _ if exact_gender => 0.45,
        _ => 0.35,
    }
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreferenceBasis {
    LiteralName,
    UniquePrecedingCandidate,
    AdjacentReflexive,
    BoundedCataphora,
    QuoteLocalNamedCandidate,
    PronounChain,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Mention {
    pub text: String,
    pub resolved_to: Option<String>,
    pub start: usize,
    pub is_pronoun: bool,
    pub confidence: f64,
    #[serde(default)]
    pub range: Option<TextRange>,
    #[serde(default)]
    pub resolution_basis: Option<CoreferenceBasis>,
    #[serde(default)]
    pub antecedent_start: Option<usize>,
}

fn sorted_counts<S: serde::Serializer>(
    counts: &HashMap<String, usize>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    counts
        .iter()
        .collect::<BTreeMap<_, _>>()
        .serialize(serializer)
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorefResult {
    pub mentions: Vec<Mention>,
    #[serde(serialize_with = "sorted_counts")]
    pub character_mentions: HashMap<String, usize>,
    /// Compatibility coverage convention; use resolution_coverage for empty input.
    pub pronoun_resolution_rate: f64,
    #[serde(default)]
    pub availability: AnalysisAvailability,
    #[serde(default)]
    pub diagnostics: Vec<String>,
    #[serde(default)]
    pub pronoun_count: usize,
    #[serde(default)]
    pub resolved_pronoun_count: usize,
    #[serde(default)]
    pub resolution_coverage: Option<f64>,
}

fn validate_inputs(
    scenes: &[&str],
    characters: &[&str],
    gender_map: Option<&HashMap<String, String>>,
) -> Result<()> {
    if scenes.len() > 10_000 || characters.len() > 256 {
        return Err(Error::InvalidInput(
            "coreference exceeds scene/cast limits".into(),
        ));
    }
    let mut bytes = 0usize;
    for scene in scenes {
        bytes = bytes
            .checked_add(scene.len())
            .filter(|n| *n <= 64 * 1024 * 1024)
            .ok_or_else(|| Error::InvalidInput("coreference source exceeds64MiB".into()))?;
        if scene.len() > 16 * 1024 * 1024 {
            return Err(Error::InvalidInput("coreference scene exceeds16MiB".into()));
        }
    }
    let mut names = BTreeSet::new();
    for character in characters {
        validate_name(character)?;
        if !names.insert(character.to_lowercase()) {
            return Err(Error::InvalidInput(
                "coreference cast names must be unique ignoring case".into(),
            ));
        }
    }
    if let Some(map) = gender_map {
        for (name, gender) in map {
            if !characters.contains(&name.as_str())
                || !["male", "female", "neutral", "unknown"].contains(&gender.as_str())
            {
                return Err(Error::InvalidInput(
                    "coreference gender hints require known names and supported labels".into(),
                ));
            }
        }
    }
    Ok(())
}
fn patterns<'a>(characters: &[&'a str]) -> Result<Vec<(&'a str, Regex)>> {
    characters
        .iter()
        .map(|&name| {
            Regex::new(&format!(r"(?i)\b{}\b", regex::escape(name)))
                .map(|pattern| (name, pattern))
                .map_err(|_| {
                    Error::InvalidInput("coreference name regex exceeds runtime limits".into())
                })
        })
        .collect()
}
fn names_in(text: &str, patterns: &[(&str, Regex)]) -> Result<Vec<Mention>> {
    let mut found = Vec::new();
    for (name, pattern) in patterns {
        for matched in pattern.find_iter(text) {
            if found.len() == MAX_COREFERENCE_MENTIONS {
                return Err(Error::InvalidInput(
                    "coreference mention candidates exceed10000".into(),
                ));
            }
            found.push(Mention {
                text: matched.as_str().into(),
                resolved_to: Some((*name).into()),
                start: matched.start(),
                is_pronoun: false,
                confidence: 0.95,
                range: None,
                resolution_basis: Some(CoreferenceBasis::LiteralName),
                antecedent_start: None,
            });
        }
    }
    found.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then_with(|| b.text.len().cmp(&a.text.len()))
            .then_with(|| a.resolved_to.cmp(&b.resolved_to))
    });
    let mut names: Vec<Mention> = Vec::new();
    for candidate in found {
        if let Some(last) = names.last_mut() {
            if last.start == candidate.start && last.text.len() == candidate.text.len() {
                if last.resolved_to != candidate.resolved_to {
                    last.resolved_to = None;
                    last.resolution_basis = None;
                    last.confidence = 0.0;
                }
                continue;
            }
            if candidate.start < last.start + last.text.len() {
                continue;
            }
        }
        names.push(candidate);
    }
    Ok(names)
}

fn inside(ranges: &[(usize, usize)], position: usize) -> bool {
    ranges
        .get(ranges.partition_point(|(_, end)| *end <= position))
        .is_some_and(|(start, end)| *start <= position && position < *end)
}
fn quote_scope(ranges: &[(usize, usize)], boundaries: &[usize], position: usize) -> Option<usize> {
    inside(ranges, position).then(|| boundaries.partition_point(|boundary| *boundary <= position))
}

fn gender_counts(
    text: &str,
    names: &[Mention],
    counts: &mut HashMap<&str, (usize, usize)>,
    quotes: &[(usize, usize)],
    groups: &HashMap<&str, usize>,
) {
    let group = |name: &Mention| name.resolved_to.as_deref().map(|name| groups[name]);
    let sentences = crate::substrate::utils::sentence_spans(text);
    let mut start = 0;
    let mut name_cursor = 0;
    for end in paragraph_regex()
        .find_iter(text)
        .map(|m| m.start())
        .chain(std::iter::once(text.len()))
    {
        let first_name = name_cursor;
        while name_cursor < names.len() && names[name_cursor].start < end {
            name_cursor += 1;
        }
        let local: Vec<_> = names[first_name..name_cursor]
            .iter()
            .filter(|name| !inside(quotes, name.start))
            .collect();
        let unique: BTreeSet<_> = local.iter().filter_map(|name| group(name)).collect();
        let whole_paragraph =
            unique.len() == 1 && local.iter().all(|name| name.resolved_to.is_some());
        for pronoun in pronoun_matcher().find_iter(&text[start..end]) {
            let position = start + pronoun.start;
            if inside(quotes, position) {
                continue;
            }
            // A paragraph naming one character attributes its later pronouns to
            // that character; otherwise only the current and previous sentence
            // may name exactly one.
            let window_start = if whole_paragraph {
                start
            } else {
                let sentence =
                    sentences.partition_point(|(_, sentence_end)| *sentence_end <= position);
                sentences
                    .get(sentence.saturating_sub(1))
                    .map_or(start, |(sentence_start, _)| (*sentence_start).max(start))
            };
            let window = &local[local.partition_point(|name| name.start < window_start)
                ..local.partition_point(|name| name.start < position)];
            let Some(first) = window.first() else {
                continue;
            };
            if !whole_paragraph && window.iter().any(|name| group(name) != group(first)) {
                continue;
            }
            let Some(name) = first.resolved_to.as_deref() else {
                continue;
            };
            if position < first.start + first.text.len() || position - first.start > 200 {
                continue;
            }
            if let Some(count) = counts.get_mut(name) {
                match pronoun_gender(pronoun.text) {
                    Gender::Male => count.0 += 1,
                    Gender::Female => count.1 += 1,
                    _ => {}
                }
            }
        }
        start = end;
    }
}
fn title_gender(token: &str) -> Option<&'static str> {
    match token.trim_end_matches('.').to_lowercase().as_str() {
        "mr" | "mister" | "sir" | "lord" | "king" | "prince" | "duke" | "baron" | "uncle" => {
            Some("male")
        }
        "mrs" | "miss" | "ms" | "madam" | "madame" | "lady" | "queen" | "princess" | "duchess"
        | "countess" | "baroness" | "aunt" | "dame" => Some("female"),
        _ => None,
    }
}
/// A cast name reduced to its name tokens: leading determiners and honorifics
/// are dropped, keeping the gender of the first gendered honorific and of the
/// first token when it is a listed given name.
struct NameCore {
    tokens: Vec<String>,
    title: Option<&'static str>,
    given: Option<&'static str>,
}
impl NameCore {
    fn parse(name: &str) -> Self {
        use crate::substrate::given_names::{GivenNameGender, given_name_gender};
        let mut title = None;
        let mut tokens = name
            .split_whitespace()
            .map(|token| token.trim_matches(|c| c == '.' || c == ',').to_lowercase())
            .filter(|token| !token.is_empty())
            .peekable();
        while let Some(token) = tokens.peek() {
            if matches!(token.as_str(), "the" | "this" | "that") {
                tokens.next();
            } else if crate::substrate::utils::is_honorific(token) || title_gender(token).is_some()
            {
                title = title.or_else(|| title_gender(token));
                tokens.next();
            } else {
                break;
            }
        }
        let given = match tokens.peek().and_then(|first| given_name_gender(first)) {
            Some(GivenNameGender::Male) => Some("male"),
            Some(GivenNameGender::Female) => Some("female"),
            _ => None,
        };
        Self {
            tokens: tokens.collect(),
            title,
            given,
        }
    }
    /// The title's gender, else the given name's.
    fn hint(&self) -> Option<&'static str> {
        self.title.or(self.given)
    }
}
/// Groups cast names that spell one character: a name joins a longer name
/// whose core tokens start with its own (`Carl` with `Carl Linstrum`) or
/// that shares its core under a title (`Bibot` with `Sergeant Bibot`).
/// Gender hints must not conflict, and a form without a gender of its own
/// (`Lebrun`, `Captain Eldridge`) never joins a female-titled one (`Madame
/// Lebrun`), since a bare surname in prose is usually the man. A group whose
/// cores are incomparable (`Peter Morrison` against `Peter Kronborg`) is
/// dissolved into singletons.
fn alias_groups<'a>(characters: &[&'a str]) -> HashMap<&'a str, usize> {
    let cores: Vec<_> = characters
        .iter()
        .map(|name| NameCore::parse(name))
        .collect();
    let prefix = |a: usize, b: usize| {
        !cores[a].tokens.is_empty() && cores[b].tokens.starts_with(&cores[a].tokens)
    };
    let agree = |a: usize, b: usize| {
        let (x, y) = (&cores[a], &cores[b]);
        x.hint().zip(y.hint()).is_none_or(|(p, q)| p == q)
            && !(x.hint().is_none() && y.title == Some("female"))
            && !(y.hint().is_none() && x.title == Some("female"))
    };
    let mut parent: Vec<usize> = (0..characters.len()).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for a in 0..characters.len() {
        for b in 0..characters.len() {
            if a != b && prefix(a, b) && agree(a, b) {
                let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                parent[ra.max(rb)] = ra.min(rb);
            }
        }
    }
    let mut members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..characters.len() {
        members.entry(root(&mut parent, i)).or_default().push(i);
    }
    let mut groups = HashMap::with_capacity(characters.len());
    let mut next = 0;
    for group in members.values() {
        let consistent = group.iter().all(|&a| {
            group
                .iter()
                .all(|&b| a == b || ((prefix(a, b) || prefix(b, a)) && agree(a, b)))
        });
        if consistent {
            for &i in group {
                groups.insert(characters[i], next);
            }
            next += 1;
        } else {
            for &i in group {
                groups.insert(characters[i], next);
                next += 1;
            }
        }
    }
    groups
}
fn finish_genders(
    counts: HashMap<&str, (usize, usize)>,
    groups: &HashMap<&str, usize>,
) -> HashMap<String, String> {
    let mut pooled: HashMap<usize, (usize, usize, Option<&'static str>)> = HashMap::new();
    for (name, (male, female)) in &counts {
        let entry = pooled.entry(groups[name]).or_default();
        entry.0 += male;
        entry.1 += female;
        entry.2 = entry.2.or(NameCore::parse(name).hint());
    }
    counts
        .into_keys()
        .map(|name| {
            let (male, female, title) = pooled[&groups[name]];
            let total = male + female;
            let gender = title.unwrap_or(if total < 3 {
                "unknown"
            } else if male as f64 / total as f64 > 0.65 {
                "male"
            } else if female as f64 / total as f64 > 0.65 {
                "female"
            } else {
                "unknown"
            });
            (name.into(), gender.into())
        })
        .collect()
}
fn stamp_ranges(text: &str, mentions: &mut [Mention]) -> Result<()> {
    let offsets = OffsetIndex::new(text)?;
    let boundary = |byte: usize| {
        offsets.utf16(byte).ok_or_else(|| {
            Error::InvalidInput("coreference range is not a character boundary".into())
        })
    };
    for mention in mentions {
        let end = mention
            .start
            .checked_add(mention.text.len())
            .ok_or_else(|| Error::InvalidInput("coreference range overflow".into()))?;
        if text.get(mention.start..end) != Some(mention.text.as_str()) {
            return Err(Error::InvalidInput(
                "coreference mention differs from original source".into(),
            ));
        }
        mention.range = Some(TextRange {
            byte_start: mention.start,
            byte_end: end,
            utf16_start: boundary(mention.start)?,
            utf16_end: boundary(end)?,
        });
    }
    Ok(())
}

fn cataphoric_candidate<'a>(
    text: &str,
    position: usize,
    token_end: usize,
    names: &'a [Mention],
    sentences: &[(usize, usize)],
    quotes: &[(usize, usize)],
    compatible: impl Fn(&str) -> bool,
) -> Option<&'a Mention> {
    let &(start, end) = sentences.get(sentences.partition_point(|(_, end)| *end <= position))?;
    if start > position {
        return None;
    }
    let first = text[start..position]
        .split_whitespace()
        .next()?
        .to_ascii_lowercase();
    if ![
        "when", "before", "after", "although", "while", "if", "once", "until",
    ]
    .contains(&first.as_str())
    {
        return None;
    }
    let comma = token_end + text[token_end..end].find(',')?;
    let main_start =
        comma + 1 + text[comma + 1..end].len() - text[comma + 1..end].trim_start().len();
    if main_start > position + 100 {
        return None;
    }
    let first = names.partition_point(|name| name.start < main_start);
    let last = names.partition_point(|name| name.start < end && name.start <= position + 100);
    let mut candidates = names[first..last].iter().filter(|name| {
        name.start + name.text.len() <= end
            && !inside(quotes, name.start)
            && name.resolved_to.as_deref().is_some_and(&compatible)
    });
    let candidate = candidates.next()?;
    (candidate.start == main_start && candidates.next().is_none()).then_some(candidate)
}

pub struct CoreferenceResolver;
impl Default for CoreferenceResolver {
    fn default() -> Self {
        Self::new()
    }
}
impl CoreferenceResolver {
    pub fn new() -> Self {
        Self
    }
    pub fn build_character_gender_map_checked(
        &self,
        scenes: &[&str],
        characters: &[&str],
    ) -> Result<HashMap<String, String>> {
        validate_inputs(scenes, characters, None)?;
        let patterns = patterns(characters)?;
        let groups = alias_groups(characters);
        let mut counts = characters.iter().map(|&name| (name, (0, 0))).collect();
        for text in scenes {
            let scopes = quotation_scopes(text)?;
            gender_counts(
                text,
                &names_in(text, &patterns)?,
                &mut counts,
                &scopes.spans,
                &groups,
            );
        }
        Ok(finish_genders(counts, &groups))
    }
    /// Co-occurrence hints are heuristic pronoun categories, not verified gender.
    pub fn build_character_gender_map(
        &self,
        scenes: &[&str],
        characters: &[&str],
    ) -> HashMap<String, String> {
        self.build_character_gender_map_checked(scenes, characters)
            .unwrap_or_default()
    }
    pub fn resolve(
        &self,
        text: &str,
        characters: &[&str],
        gender_map: Option<&HashMap<String, String>>,
    ) -> CorefResult {
        self.resolve_checked(text, characters, gender_map)
            .unwrap_or_else(|error| CorefResult {
                mentions: Vec::new(),
                character_mentions: HashMap::new(),
                pronoun_resolution_rate: 0.0,
                availability: AnalysisAvailability::Unavailable,
                diagnostics: vec![error.to_string()],
                pronoun_count: 0,
                resolved_pronoun_count: 0,
                resolution_coverage: None,
            })
    }
    pub fn resolve_checked(
        &self,
        text: &str,
        characters: &[&str],
        gender_map: Option<&HashMap<String, String>>,
    ) -> Result<CorefResult> {
        validate_inputs(&[text], characters, gender_map)?;
        let name_mentions = names_in(text, &patterns(characters)?)?;
        let scopes = quotation_scopes(text)?;
        let quote_ranges = scopes.spans;
        let quote_boundaries = scopes.boundaries;
        let mut diagnostics = scopes.diagnostics;
        let groups = alias_groups(characters);
        let group = |name: &Mention| name.resolved_to.as_deref().map(|name| groups[name]);
        let owned;
        let gender_map = if let Some(map) = gender_map {
            map
        } else {
            let mut counts = characters.iter().map(|&name| (name, (0, 0))).collect();
            gender_counts(text, &name_mentions, &mut counts, &quote_ranges, &groups);
            owned = finish_genders(counts, &groups);
            &owned
        };
        // Caller hints are pooled per alias group; disagreeing hints leave the
        // group unknown.
        let mut group_genders = vec![None::<Gender>; characters.len()];
        for &name in characters {
            let hint = gender_from_str(gender_map.get(name).map_or("unknown", String::as_str));
            if hint == Gender::Unknown {
                continue;
            }
            let slot = &mut group_genders[groups[name]];
            *slot = match *slot {
                None => Some(hint),
                Some(agreed) if agreed == hint => Some(hint),
                Some(_) => Some(Gender::Unknown),
            };
        }
        let group_gender = |name: &str| group_genders[groups[name]].unwrap_or(Gender::Unknown);
        let breaks: Vec<_> = paragraph_regex()
            .find_iter(text)
            .map(|m| m.start())
            .collect();
        let same_paragraph = |a: usize, b: usize| {
            let lo = a.min(b);
            let hi = a.max(b);
            breaks
                .get(breaks.partition_point(|brk| *brk <= lo))
                .is_none_or(|brk| *brk >= hi)
        };
        let sentences = crate::substrate::utils::sentence_spans(text);
        let mut pronouns: Vec<Mention> = Vec::new();
        let mut chain_depths: Vec<u8> = Vec::new();
        // Narrative and quoted pronoun chains are tracked apart so a quoted
        // pronoun does not sever the narrative chain around it.
        let mut previous_pronouns = [None::<usize>; 3];
        let mut previous_quoted = [None::<usize>; 3];
        for token in pronoun_matcher().find_iter(text) {
            let end_names = name_mentions.partition_point(|name| name.start <= token.start);
            if end_names > 0
                && name_mentions[end_names - 1].start + name_mentions[end_names - 1].text.len()
                    >= token.end
            {
                continue;
            }
            if name_mentions.len() + pronouns.len() == MAX_COREFERENCE_MENTIONS {
                return Err(Error::InvalidInput(
                    "coreference mention candidates exceed10000".into(),
                ));
            }
            let mut mention = Mention {
                text: token.text.into(),
                resolved_to: None,
                start: token.start,
                is_pronoun: true,
                confidence: 0.0,
                range: None,
                resolution_basis: None,
                antecedent_start: None,
            };
            let gender = pronoun_gender(token.text);
            let gender_index = match gender {
                Gender::Male => 0,
                Gender::Female => 1,
                _ => 2,
            };
            let mut chain_depth = 0;
            {
                let scope = quote_scope(&quote_ranges, &quote_boundaries, token.start);
                let kind = pronoun_kind(token.text);
                let compatible = |name: &str| {
                    let candidate = group_gender(name);
                    if gender == Gender::Neutral {
                        // `they` is overwhelmingly plural in prose; a single name
                        // may bind only when the caller declared it neutral.
                        candidate == Gender::Neutral
                    } else {
                        candidate == Gender::Unknown || candidate == gender
                    }
                };
                let start = name_mentions
                    .partition_point(|name| name.start < token.start.saturating_sub(300));
                let end = name_mentions.partition_point(|name| name.start < token.start);
                let mut candidates = BTreeMap::new();
                let mut ambiguous = false;
                for name in &name_mentions[start..end] {
                    if name.start + name.text.len() > token.start
                        || quote_scope(&quote_ranges, &quote_boundaries, name.start) != scope
                    {
                        continue;
                    }
                    if let Some(character) = name.resolved_to.as_deref() {
                        if compatible(character) {
                            candidates.insert(groups[character], name);
                        }
                    } else {
                        ambiguous = true;
                    }
                }
                let adjacent = if kind == PronounKind::Reflexive {
                    candidates
                        .values()
                        .find(|name| {
                            text[name.start + name.text.len()..token.start]
                                .chars()
                                .all(char::is_whitespace)
                        })
                        .copied()
                } else {
                    None
                };
                if !ambiguous && (candidates.len() == 1 || adjacent.is_some()) {
                    let name = adjacent
                        .unwrap_or_else(|| *candidates.values().next().expect("one candidate"));
                    let character = name.resolved_to.as_ref().expect("resolved candidate");
                    mention.resolved_to = Some(character.clone());
                    mention.antecedent_start = Some(name.start);
                    mention.resolution_basis = Some(if adjacent.is_some() {
                        CoreferenceBasis::AdjacentReflexive
                    } else if scope.is_some() {
                        CoreferenceBasis::QuoteLocalNamedCandidate
                    } else {
                        CoreferenceBasis::UniquePrecedingCandidate
                    });
                    mention.confidence = if gender == Gender::Neutral {
                        0.40
                    } else {
                        compute_confidence(
                            kind,
                            group_gender(character) == gender,
                            same_paragraph(name.start, token.start),
                            token.start - name.start,
                            false,
                        )
                    };
                } else if !ambiguous
                    && candidates.is_empty()
                    && scope.is_none()
                    && kind != PronounKind::Reflexive
                    && let Some(name) = cataphoric_candidate(
                        text,
                        token.start,
                        token.end,
                        &name_mentions,
                        &sentences,
                        &quote_ranges,
                        compatible,
                    )
                {
                    mention.resolved_to = name.resolved_to.clone();
                    mention.antecedent_start = Some(name.start);
                    mention.confidence = 0.30;
                    mention.resolution_basis = Some(CoreferenceBasis::BoundedCataphora);
                }
                let previous = if scope.is_none() {
                    previous_pronouns[gender_index]
                } else {
                    previous_quoted[gender_index]
                };
                if mention.resolved_to.is_none()
                    && !ambiguous
                    && candidates.is_empty()
                    && kind != PronounKind::Reflexive
                    && let Some(previous) = previous
                {
                    let antecedent = &pronouns[previous];
                    let first_name =
                        name_mentions.partition_point(|name| name.start < antecedent.start);
                    let next_quote = quote_boundaries
                        .get(quote_boundaries.partition_point(|start| *start <= antecedent.start));
                    let competing_name = name_mentions[first_name..end].iter().any(|name| {
                        name.resolved_to.as_deref().is_none_or(|character| {
                            group(name) != group(antecedent) && compatible(character)
                        })
                    });
                    if antecedent.resolved_to.is_some()
                        && chain_depths[previous] < MAX_COREFERENCE_CHAIN_DEPTH
                        && token.start - antecedent.start <= 300
                        && same_paragraph(antecedent.start, token.start)
                        && quote_scope(&quote_ranges, &quote_boundaries, antecedent.start) == scope
                        && (scope.is_none() || next_quote.is_none_or(|start| *start >= token.start))
                        && !competing_name
                    {
                        mention.resolved_to = antecedent.resolved_to.clone();
                        mention.antecedent_start = Some(antecedent.start);
                        mention.resolution_basis = Some(CoreferenceBasis::PronounChain);
                        mention.confidence = 0.30;
                        chain_depth = chain_depths[previous] + 1;
                    }
                }
            }
            if quote_scope(&quote_ranges, &quote_boundaries, token.start).is_none() {
                previous_pronouns[gender_index] = Some(pronouns.len());
            } else {
                previous_quoted[gender_index] = Some(pronouns.len());
            }
            chain_depths.push(chain_depth);
            pronouns.push(mention);
        }
        let pronoun_count = pronouns.len();
        let resolved_pronoun_count = pronouns
            .iter()
            .filter(|mention| mention.resolved_to.is_some())
            .count();
        let resolution_coverage =
            (pronoun_count > 0).then(|| resolved_pronoun_count as f64 / pronoun_count as f64);
        let mut mentions = name_mentions;
        mentions.extend(pronouns);
        mentions.sort_by_key(|mention| mention.start);
        stamp_ranges(text, &mut mentions)?;
        let mut character_mentions = characters
            .iter()
            .map(|&name| (name.into(), 0))
            .collect::<HashMap<_, _>>();
        for mention in &mentions {
            if let Some(character) = &mention.resolved_to {
                *character_mentions
                    .get_mut(character)
                    .expect("resolved cast identity") += 1;
            }
        }
        diagnostics.push("Literal-name, quote-local named antecedents and bounded scope-local pronoun-chain hypotheses only; confidence tiers and resolution coverage are not semantic accuracy probabilities. Dialogue speakers and ambiguous antecedents remain unassessed; cataphora is limited to a subordinate clause with one visible main-clause name.".into());
        Ok(CorefResult {
            mentions,
            character_mentions,
            pronoun_resolution_rate: resolution_coverage.unwrap_or(1.0),
            availability: AnalysisAvailability::Partial,
            diagnostics,
            pronoun_count,
            resolved_pronoun_count,
            resolution_coverage,
        })
    }
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AuthorCoreferenceLink {
    pub id: String,
    pub mention: AuthorSource,
    pub antecedent: AuthorSource,
    pub character_id: String,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedAuthorCoreference {
    pub id: String,
    pub character_id: String,
    pub mention: Evidence,
    pub antecedent: Evidence,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthorCoreferenceResult {
    pub document_id: String,
    pub availability: AnalysisAvailability,
    pub diagnostics: Vec<String>,
    pub links: Vec<ResolvedAuthorCoreference>,
}

pub fn resolve_author_coreference(
    manuscript: &Manuscript,
    links: &[AuthorCoreferenceLink],
) -> Result<AuthorCoreferenceResult> {
    manuscript.validate()?;
    if links.len() > MAX_COREFERENCE_MENTIONS {
        return Err(Error::InvalidInput(
            "author coreference exceeds10000 links".into(),
        ));
    }
    let characters: BTreeSet<_> = manuscript
        .known_characters
        .iter()
        .map(String::as_str)
        .collect();
    let mut ids = BTreeSet::new();
    let mut spans = BTreeSet::new();
    let mut assignments = BTreeMap::new();
    let mut bytes = 0usize;
    let mut sources = Vec::with_capacity(links.len() * 2);
    for link in links {
        validate_id(&link.id)?;
        validate_id(&link.character_id)?;
        if link.mention.scene_id == link.antecedent.scene_id
            && link.mention.utf16_start == link.antecedent.utf16_start
            && link.mention.utf16_end == link.antecedent.utf16_end
        {
            return Err(Error::InvalidInput(
                "author coreference requires a distinct antecedent span".into(),
            ));
        }
        for endpoint in [&link.mention, &link.antecedent] {
            let key = (
                endpoint.scene_id.as_str(),
                endpoint.utf16_start,
                endpoint.utf16_end,
            );
            if assignments
                .insert(key, link.character_id.as_str())
                .is_some_and(|previous| previous != link.character_id)
            {
                return Err(Error::InvalidInput(
                    "author coreference assigns conflicting identities to one source span".into(),
                ));
            }
        }
        if !characters.contains(link.character_id.as_str())
            || !ids.insert(&link.id)
            || !spans.insert((
                &link.mention.scene_id,
                link.mention.utf16_start,
                link.mention.utf16_end,
            ))
        {
            return Err(Error::InvalidInput(
                "author coreference requires known identities and unique IDs/mention spans".into(),
            ));
        }
        bytes = bytes
            .checked_add(
                link.id.len()
                    + link.character_id.len()
                    + link.mention.scene_id.len()
                    + link.mention.quote.len()
                    + link.antecedent.scene_id.len()
                    + link.antecedent.quote.len(),
            )
            .filter(|n| *n <= MAX_AUTHOR_COREFERENCE_BYTES)
            .ok_or_else(|| Error::InvalidInput("author coreference metadata exceeds4MiB".into()))?;
        sources.extend([&link.mention, &link.antecedent]);
    }
    let mut evidence = validate_sources(manuscript, &sources)?.into_iter();
    let mut resolved: Vec<_> = links
        .iter()
        .map(|link| ResolvedAuthorCoreference {
            id: link.id.clone(),
            character_id: link.character_id.clone(),
            mention: evidence.next().expect("mention source validated"),
            antecedent: evidence.next().expect("antecedent source validated"),
        })
        .collect();
    resolved.sort_by(|a, b| {
        a.mention
            .scene_id
            .cmp(&b.mention.scene_id)
            .then_with(|| a.mention.range.byte_start.cmp(&b.mention.range.byte_start))
            .then_with(|| a.mention.range.byte_end.cmp(&b.mention.range.byte_end))
            .then_with(|| a.id.cmp(&b.id))
    });
    Ok(AuthorCoreferenceResult { document_id: manuscript.document_id.clone(), availability: AnalysisAvailability::Complete, diagnostics: vec!["Author-supplied identity links only; exact sources locate annotations without proving their semantic interpretation. Nested mentions and explicit cataphora are supported; no chronology is inferred.".into()], links: resolved })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let resolver = CoreferenceResolver::new();
        let result = resolver.resolve("", &[], None);
        assert!(result.mentions.is_empty());
        assert_eq!(result.pronoun_resolution_rate, 1.0);
    }

    #[test]
    fn test_basic_he_resolution() {
        let resolver = CoreferenceResolver::new();
        let text = "John walked to the store. He bought some milk.";
        let mut gender_map = HashMap::new();
        gender_map.insert("John".to_string(), "male".to_string());
        let result = resolver.resolve(text, &["John"], Some(&gender_map));

        let he_mention = result
            .mentions
            .iter()
            .find(|m| m.is_pronoun && m.text.to_lowercase() == "he")
            .expect("should find 'He' pronoun");
        assert_eq!(he_mention.resolved_to.as_deref(), Some("John"));
        assert!(he_mention.confidence > 0.0);
        assert!(result.pronoun_resolution_rate > 0.0);
    }

    #[test]
    fn test_gender_inference() {
        let resolver = CoreferenceResolver::new();
        let scenes = vec![
            "John picked up his bag. He walked to the door. His coat was heavy.",
            "John said he would return. He promised his friend.",
        ];
        let map = resolver.build_character_gender_map(&scenes, &["John"]);
        assert_eq!(map.get("John").map(|s| s.as_str()), Some("male"));
    }

    #[test]
    fn test_paragraph_boundary() {
        let resolver = CoreferenceResolver::new();
        let text = "John entered the room.\n\nMary smiled. She waved.";
        let mut gender_map = HashMap::new();
        gender_map.insert("John".to_string(), "male".to_string());
        gender_map.insert("Mary".to_string(), "female".to_string());
        let result = resolver.resolve(text, &["John", "Mary"], Some(&gender_map));

        let she_mention = result
            .mentions
            .iter()
            .find(|m| m.is_pronoun && m.text.to_lowercase() == "she")
            .expect("should find 'She' pronoun");
        assert_eq!(she_mention.resolved_to.as_deref(), Some("Mary"));
    }

    #[test]
    fn test_they_them_neutral_resolution() {
        let resolver = CoreferenceResolver::new();
        let text = "Alex walked in. They sat down.";
        let mut gender_map = HashMap::new();
        gender_map.insert("Alex".to_string(), "neutral".to_string());
        let result = resolver.resolve(text, &["Alex"], Some(&gender_map));

        let they_mention = result
            .mentions
            .iter()
            .find(|m| m.is_pronoun && m.text.to_lowercase() == "they")
            .expect("should find 'They' pronoun");
        assert_eq!(they_mention.resolved_to.as_deref(), Some("Alex"));
        assert!((they_mention.confidence - 0.4).abs() < f64::EPSILON);

        for hint in ["unknown", "male"] {
            gender_map.insert("Alex".to_string(), hint.to_string());
            let result = resolver.resolve(text, &["Alex"], Some(&gender_map));
            assert!(
                result
                    .mentions
                    .iter()
                    .filter(|m| m.is_pronoun)
                    .all(|m| m.resolved_to.is_none()),
                "`they` must not bind to a {hint} character"
            );
        }
        let result = resolver.resolve(text, &["Alex"], None);
        assert_eq!(result.resolved_pronoun_count, 0);
    }

    #[test]
    fn gender_is_inferred_from_sentence_local_evidence_in_multi_name_paragraphs() {
        // Two names share the paragraph, so only the current/previous-sentence
        // window can attribute each `He` to Carlo. Neither name is in the
        // given-name lexicon, so the text alone decides.
        let text = "Carlo sat down. He frowned. Zelda stood. Carlo left. He ran. \
                    Zelda waited. Carlo paused. He sighed.";
        let resolver = CoreferenceResolver::new();
        let map = resolver
            .build_character_gender_map_checked(&[text], &["Carlo", "Zelda"])
            .unwrap();
        assert_eq!(map["Carlo"], "male");
        assert_eq!(map["Zelda"], "unknown");
        // A window naming two characters attributes nothing.
        let mixed = "Carlo met Zelda. He smiled. Carlo met Zelda. He sat. \
                     Carlo met Zelda. He left.";
        let map = resolver
            .build_character_gender_map_checked(&[mixed], &["Carlo", "Zelda"])
            .unwrap();
        assert_eq!(map["Carlo"], "unknown");
    }

    #[test]
    fn honorifics_in_cast_names_fix_gender_and_disambiguate_pronouns() {
        let resolver = CoreferenceResolver::new();
        let text = "Mrs. Kronborg watched Peter. He left.";
        let result = resolver
            .resolve_checked(text, &["Mrs. Kronborg", "Peter"], None)
            .unwrap();
        let pronoun = result.mentions.iter().find(|m| m.is_pronoun).unwrap();
        assert_eq!(pronoun.resolved_to.as_deref(), Some("Peter"));
        let map = resolver
            .build_character_gender_map_checked(&[text], &["Mrs. Kronborg", "Peter", "Sir Henry"])
            .unwrap();
        assert_eq!(map["Mrs. Kronborg"], "female");
        assert_eq!(map["Sir Henry"], "male");
        // No pronoun evidence, but a listed given name.
        assert_eq!(map["Peter"], "male");
    }

    #[test]
    fn alias_groups_join_given_names_and_titles_but_dissolve_conflicts() {
        let groups = alias_groups(&[
            "Carl",
            "Carl Linstrum",
            "Bibot",
            "Sergeant Bibot",
            "Peter",
            "Peter Morrison",
            "Peter Kronborg",
            "Bennet",
            "Mr. Bennet",
            "Mrs. Bennet",
            "Rachel",
            "Mrs. Rachel",
            "Rachel Lynde",
            "Lebrun",
            "Madame Lebrun",
            "Captain Eldridge",
            "Mrs. Eldridge",
            "Oak",
        ]);
        assert_eq!(groups["Carl"], groups["Carl Linstrum"]);
        assert_eq!(groups["Bibot"], groups["Sergeant Bibot"]);
        assert_ne!(groups["Carl"], groups["Bibot"]);
        // Two incomparable surnames share "Peter": nobody is merged.
        assert_ne!(groups["Peter"], groups["Peter Morrison"]);
        assert_ne!(groups["Peter Morrison"], groups["Peter Kronborg"]);
        // A bare surname reads as the man: it joins Mr. but not Mrs. Bennet.
        assert_ne!(groups["Mr. Bennet"], groups["Mrs. Bennet"]);
        assert_eq!(groups["Bennet"], groups["Mr. Bennet"]);
        // A female given name agrees with a female title.
        assert_eq!(groups["Rachel"], groups["Mrs. Rachel"]);
        assert_eq!(groups["Rachel"], groups["Rachel Lynde"]);
        // Neither a bare surname nor an ungendered title joins a Madame/Mrs.
        assert_ne!(groups["Lebrun"], groups["Madame Lebrun"]);
        assert_ne!(groups["Captain Eldridge"], groups["Mrs. Eldridge"]);
        assert_ne!(groups["Oak"], groups["Carl"]);
    }

    #[test]
    fn aliases_of_one_character_do_not_block_each_other_and_pool_gender() {
        let resolver = CoreferenceResolver::new();
        let text = "Carl Linstrum came in and Carl sat. He smiled. She waited.";
        let result = resolver
            .resolve_checked(text, &["Carl", "Carl Linstrum"], None)
            .unwrap();
        let pronoun = |word: &str| {
            result
                .mentions
                .iter()
                .find(|m| m.is_pronoun && m.text == word)
                .unwrap()
                .resolved_to
                .clone()
        };
        // Two visible mentions, one character: the nearest alias wins.
        assert_eq!(pronoun("He").as_deref(), Some("Carl"));
        // The given name makes the whole group male.
        assert_eq!(pronoun("She"), None);
        let mut hints = HashMap::new();
        hints.insert("Carl Linstrum".to_string(), "female".to_string());
        let result = resolver
            .resolve_checked(text, &["Carl", "Carl Linstrum"], Some(&hints))
            .unwrap();
        let she = result
            .mentions
            .iter()
            .find(|m| m.is_pronoun && m.text == "She")
            .unwrap();
        // A caller hint on one alias applies to the group.
        assert_eq!(she.resolved_to.as_deref(), Some("Carl"));
    }

    #[test]
    fn narrative_pronoun_chain_survives_intervening_dialogue() {
        let resolver = CoreferenceResolver::new();
        // "She waited" sits inside the 300-byte window; the later pronouns
        // fall outside it and can only chain.
        let filler = "The wind blew across the empty street and the lamps swung. ".repeat(4);
        let text = format!(
            "Alexandra stood by the pole. {filler}She waited. \"Come down, Emil! She is cold,\" she called. She shivered."
        );
        let result = resolver
            .resolve_checked(&text, &["Alexandra", "Emil"], None)
            .unwrap();
        let pronouns: Vec<_> = result
            .mentions
            .iter()
            .filter(|m| m.is_pronoun)
            .map(|m| {
                (
                    m.text.as_str(),
                    m.resolved_to.as_deref(),
                    m.resolution_basis,
                )
            })
            .collect();
        assert_eq!(pronouns[0].1, Some("Alexandra"));
        // The quoted "She" is somebody else's and stays unresolved.
        assert_eq!(pronouns[1], ("She", None, None));
        // Both narrative pronouns after the quote chain back to the narrative
        // antecedent; the quote and Emil (male) do not sever the chain.
        assert_eq!(
            pronouns[2],
            (
                "she",
                Some("Alexandra"),
                Some(CoreferenceBasis::PronounChain)
            )
        );
        assert_eq!(
            pronouns[3],
            (
                "She",
                Some("Alexandra"),
                Some(CoreferenceBasis::PronounChain)
            )
        );
    }

    #[test]
    fn test_possessive_resolution() {
        let resolver = CoreferenceResolver::new();
        let text = "Mary put down her book. John picked up his sword.";
        let mut gender_map = HashMap::new();
        gender_map.insert("Mary".to_string(), "female".to_string());
        gender_map.insert("John".to_string(), "male".to_string());
        let result = resolver.resolve(text, &["Mary", "John"], Some(&gender_map));

        let her_mention = result
            .mentions
            .iter()
            .find(|m| m.is_pronoun && m.text.to_lowercase() == "her")
            .expect("should find 'her' pronoun");
        assert_eq!(her_mention.resolved_to.as_deref(), Some("Mary"));
        assert!((her_mention.confidence - 0.40).abs() < f64::EPSILON);

        let his_mention = result
            .mentions
            .iter()
            .find(|m| m.is_pronoun && m.text.to_lowercase() == "his")
            .expect("should find 'his' pronoun");
        assert_eq!(his_mention.resolved_to.as_deref(), Some("John"));
        assert!((his_mention.confidence - 0.40).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reflexive_resolution() {
        let resolver = CoreferenceResolver::new();
        let text = "John looked at himself in the mirror.";
        let mut gender_map = HashMap::new();
        gender_map.insert("John".to_string(), "male".to_string());
        let result = resolver.resolve(text, &["John"], Some(&gender_map));

        let himself_mention = result
            .mentions
            .iter()
            .find(|m| m.is_pronoun && m.text.to_lowercase() == "himself")
            .expect("should find 'himself' pronoun");
        assert_eq!(himself_mention.resolved_to.as_deref(), Some("John"));
        assert!((himself_mention.confidence - 0.40).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cataphoric_resolution() {
        let resolver = CoreferenceResolver::new();
        let text = "When he arrived, John was tired.";
        let mut gender_map = HashMap::new();
        gender_map.insert("John".to_string(), "male".to_string());
        let result = resolver.resolve(text, &["John"], Some(&gender_map));

        let he_mention = result
            .mentions
            .iter()
            .find(|m| m.is_pronoun && m.text.to_lowercase() == "he")
            .expect("should find 'he' pronoun");
        assert_eq!(he_mention.resolved_to.as_deref(), Some("John"));
        assert!((he_mention.confidence - 0.30).abs() < f64::EPSILON);
    }

    #[test]
    fn test_confidence_tiers() {
        let resolver = CoreferenceResolver::new();
        // Same paragraph, close distance, exact gender match -> 0.85
        let text = "John smiled. He laughed.";
        let mut gender_map = HashMap::new();
        gender_map.insert("John".to_string(), "male".to_string());
        let result = resolver.resolve(text, &["John"], Some(&gender_map));

        let he_mention = result
            .mentions
            .iter()
            .find(|m| m.is_pronoun && m.text.to_lowercase() == "he")
            .expect("should find 'He' pronoun");
        assert_eq!(he_mention.resolved_to.as_deref(), Some("John"));
        assert!((he_mention.confidence - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_name_mention_confidence() {
        let resolver = CoreferenceResolver::new();
        let text = "John walked to the store.";
        let mut gender_map = HashMap::new();
        gender_map.insert("John".to_string(), "male".to_string());
        let result = resolver.resolve(text, &["John"], Some(&gender_map));

        let name_mention = result
            .mentions
            .iter()
            .find(|m| !m.is_pronoun && m.text == "John")
            .expect("should find 'John' name mention");
        assert!((name_mention.confidence - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cross_paragraph_confidence() {
        let resolver = CoreferenceResolver::new();
        let text = "John entered the room.\n\nHe sat down.";
        let mut gender_map = HashMap::new();
        gender_map.insert("John".to_string(), "male".to_string());
        let result = resolver.resolve(text, &["John"], Some(&gender_map));

        let he_mention = result
            .mentions
            .iter()
            .find(|m| m.is_pronoun && m.text.to_lowercase() == "he")
            .expect("should find 'He' pronoun");
        assert_eq!(he_mention.resolved_to.as_deref(), Some("John"));
        // Cross paragraph, exact gender, distance ~24 chars -> 0.70
        assert!((he_mention.confidence - 0.70).abs() < f64::EPSILON);
    }
}
