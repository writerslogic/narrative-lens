use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::OnceLock;

use crate::onnx;
use crate::onnx::{
    checked::{ModelFailure, ModelKind},
    model::ModelIdentity,
};
use serde::{Deserialize, Serialize};

pub const MAX_NER_TEXTS: usize = 10_000;
pub const MAX_NER_TEXT_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_NER_TOTAL_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_NER_WINDOWS: usize = 100_000;
pub const MAX_NER_ENTITIES: usize = 10_000;
pub const MAX_PERSON_TOKENS: usize = 250_000;
pub const MAX_PERSON_MARKERS: usize = 100_000;
pub const MAX_PERSON_CANDIDATES: usize = 10_000;
type NameSignals = [usize; 4];

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NerBatchResult {
    pub entities: Vec<onnx::NerSpans>,
    pub model_identity: Option<ModelIdentity>,
    pub model_generation: u64,
    pub windows: usize,
}

/// Set the explicit model root containing `ner-onnx/`.
/// Replaces or retries the native model; a no-op on wasm.
pub fn set_ner_model_directory(path: &str) {
    onnx::set_ner_model_directory(path);
}

/// Byte windows are a heuristic budget, not a tokenizer-token guarantee. Checked
/// model profiles enforce their actual token limit and propagate failures.
const NER_WINDOW_BYTES: usize = 1500;

/// Run the supplied BIO token classifier over `text`, returning (entity_text, label, start_byte,
/// end_byte) in the **full-text** coordinate frame.
///
/// Long inputs are split into ≤[`NER_WINDOW_BYTES`] byte windows (whitespace
/// boundaries are preferred) and each window is tagged separately, with
/// offsets shifted back. This is what lets NER evaluate an entire scene: the
/// earlier fixed-char cap dropped every entity past the model's token limit,
/// leaving long scenes with no person/place evidence at all.
///
/// Delegates to the active [`crate::onnx::InferenceBackend`] (ONNX Runtime natively,
/// `tract` on wasm); returns an empty Vec when no model is available.
pub fn ner_extract_entities(text: &str) -> Vec<(String, String, usize, usize)> {
    ner_extract_entities_batch(&[text])
        .into_iter()
        .next()
        .unwrap_or_default()
}

/// How many NER windows to feed the model per padded ONNX run. Windows are
/// near-uniform (~[`NER_WINDOW_BYTES`]), so padding waste within a batch is
/// small while one matmul replaces this many sequential inferences.
const NER_BATCH: usize = 16;

/// Tag entities across many texts with as few padded ONNX runs as possible.
///
/// Every text is split into ≤[`NER_WINDOW_BYTES`] windows, the windows from all
/// texts are pooled, run through the model [`NER_BATCH`] at a time, and the
/// spans are regrouped per text with offsets shifted back to that text's frame.
/// This turns the old one-inference-per-window cost (minutes on a real
/// manuscript) into a handful of batched runs. Returns one entity list per
/// input text, in order.
pub fn ner_extract_entities_batch(texts: &[&str]) -> Vec<Vec<(String, String, usize, usize)>> {
    ner_extract_entities_batch_checked(texts)
        .map(|result| result.entities)
        .unwrap_or_else(|_| {
            if texts.len() <= MAX_NER_TEXTS {
                vec![Vec::new(); texts.len()]
            } else {
                Vec::new()
            }
        })
}

pub fn ner_extract_entities_batch_checked(texts: &[&str]) -> Result<NerBatchResult, ModelFailure> {
    ner_extract_entities_batch_with_backend_checked(onnx::backend(), texts)
}

pub fn ner_extract_entities_batch_with_backend_checked(
    backend: &dyn onnx::InferenceBackend,
    texts: &[&str],
) -> Result<NerBatchResult, ModelFailure> {
    extract_checked(backend, texts, onnx::model_generation)
}

fn extract_checked(
    backend: &dyn onnx::InferenceBackend,
    texts: &[&str],
    generation_source: impl Fn() -> u64,
) -> Result<NerBatchResult, ModelFailure> {
    if texts.len() > MAX_NER_TEXTS {
        return Err(ModelFailure::execution("ner_text_count_limit"));
    }
    let mut total = 0usize;
    for text in texts {
        total += text.len();
        if text.len() > MAX_NER_TEXT_BYTES || total > MAX_NER_TOTAL_BYTES {
            return Err(ModelFailure::execution("ner_text_bytes_limit"));
        }
    }
    let generation = generation_source();
    // window k belongs to text `text_idx` and starts at byte `start` within it.
    struct Win {
        text_idx: usize,
        start: usize,
    }
    let mut wins: Vec<Win> = Vec::new();
    let mut win_strs: Vec<&str> = Vec::new();
    for (ti, text) in texts.iter().enumerate() {
        if text.trim().is_empty() {
            continue;
        }
        if text.len() <= NER_WINDOW_BYTES {
            wins.push(Win {
                text_idx: ti,
                start: 0,
            });
            win_strs.push(text);
        } else {
            for (start, end) in ner_windows(text, NER_WINDOW_BYTES) {
                wins.push(Win {
                    text_idx: ti,
                    start,
                });
                win_strs.push(&text[start..end]);
            }
        }
        if wins.len() > MAX_NER_WINDOWS {
            return Err(ModelFailure::execution("ner_window_limit"));
        }
    }

    let mut out: Vec<Vec<(String, String, usize, usize)>> = vec![Vec::new(); texts.len()];
    if win_strs.is_empty() {
        return Ok(NerBatchResult {
            entities: out,
            model_identity: None,
            model_generation: generation,
            windows: 0,
        });
    }
    let identity = backend.model_identity(ModelKind::Ner)?;
    identity.validate(ModelKind::Ner)?;
    let mut entity_count = 0usize;
    for base in (0..win_strs.len()).step_by(NER_BATCH) {
        let end = (base + NER_BATCH).min(win_strs.len());
        for (k, ents) in onnx::checked::entities(backend, &win_strs[base..end])?
            .into_iter()
            .enumerate()
        {
            let w = &wins[base + k];
            for (name, label, s, e) in ents {
                entity_count += 1;
                if entity_count > MAX_NER_ENTITIES {
                    return Err(ModelFailure::execution("ner_entity_count_limit"));
                }
                out[w.text_idx].push((name, label, w.start + s, w.start + e));
            }
        }
    }
    if generation_source() != generation {
        return Err(ModelFailure::execution("model_replaced_during_ner"));
    }
    for entities in &mut out {
        entities.sort_unstable_by(|a, b| {
            a.2.cmp(&b.2)
                .then_with(|| a.3.cmp(&b.3))
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.0.cmp(&b.0))
        });
        entities.dedup();
    }
    Ok(NerBatchResult {
        entities: out,
        model_identity: Some(identity),
        model_generation: generation,
        windows: wins.len(),
    })
}

/// Partition `text` into covering windows of at most `max` bytes, breaking on
/// whitespace where possible and always on a char boundary. Guarantees forward
/// progress so the loop terminates even on a single over-long token.
fn ner_windows(text: &str, max: usize) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end =
            crate::substrate::text::floor_char_boundary(text, (start + max).min(text.len()));
        if end < text.len() {
            // Back up to the last whitespace so an entity isn't split across windows.
            if let Some(ws) = text[start..end].rfind(char::is_whitespace)
                && ws > 0
            {
                end = start
                    + ws
                    + text[start + ws..]
                        .chars()
                        .next()
                        .expect("matched whitespace")
                        .len_utf8();
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
// Five-phase heuristic character candidate extraction
// ---------------------------------------------------------------------------

fn capitalized_name_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(r"\b\p{Lu}[\p{L}\p{M}'’-]*(?:[ \t]+\p{Lu}[\p{L}\p{M}'’-]*){0,2}\b")
            .expect("capitalized-name regex")
    })
}

/// Phase 1 output: the per-name signal counts plus the set of surfaces the NER
/// tagger labeled as a *non-person* entity (LOC/ORG/MISC) and never
/// as a person. These are used downstream to veto place/organization names
/// ("New York", "Associates") that the capitalization/regex signals would
/// otherwise admit as characters.
struct Phase1 {
    candidates: HashMap<String, NameSignals>,
    /// Title-cased surfaces (and their component words) NER labeled non-person.
    non_person: HashSet<String>,
    /// Title-cased surfaces in an attribution frame; this heuristic exempts a
    /// candidate from lexical vetoes without proving personhood.
    protected: HashSet<String>,
}

/// Phase 1: Broad candidate collection from all signal sources.
fn candidate_name(surface: &str) -> crate::Result<String> {
    if surface.len() > 128 {
        return Err(crate::Error::InvalidInput(
            "person candidate name exceeds128bytes".into(),
        ));
    }
    let name = title_case(surface.trim());
    crate::manuscript::config::validate_name(&name)?;
    Ok(name)
}
fn add_signal(
    candidates: &mut HashMap<String, NameSignals>,
    surface: &str,
    signal: usize,
    count: usize,
) -> crate::Result<()> {
    let name = candidate_name(surface)?;
    candidates.entry(name).or_insert([0; 4])[signal] += count;
    if candidates.len() > MAX_PERSON_CANDIDATES {
        return Err(crate::Error::InvalidInput(
            "person candidate count exceeds limit".into(),
        ));
    }
    Ok(())
}
fn phase1_collect(scenes: &[String], model_entities: &[onnx::NerSpans]) -> crate::Result<Phase1> {
    let mut candidates = HashMap::new();
    let mut per_hits = HashSet::new();
    let mut non_person = HashSet::new();
    for scene_entities in model_entities {
        for (name, label, _, _) in scene_entities {
            let norm = candidate_name(name)?;
            if label == "PER" {
                add_signal(&mut candidates, name, 0, 1)?;
                per_hits.insert(norm);
            } else {
                non_person.insert(norm.clone());
                for word in norm.split_whitespace() {
                    if word.chars().count() >= 3 {
                        non_person.insert(title_case(word));
                    }
                }
            }
        }
    }
    non_person.retain(|surface| !per_hits.contains(surface));
    // Segment each scene once: the attribution/capitalization regex pass and
    // the capitalized-name pass below both skip sentence-initial tokens.
    let sentence_starts: Vec<HashSet<usize>> = scenes
        .iter()
        .map(|scene| crate::substrate::nlp::sentence_starts(scene))
        .collect();
    let evidence = crate::substrate::nlp::regex_name_evidence(scenes, &sentence_starts);
    for (name, count) in evidence.names {
        add_signal(&mut candidates, &name, 1, count)?;
    }
    for (scene, starts) in scenes.iter().zip(&sentence_starts) {
        for matched in capitalized_name_re().find_iter(scene) {
            if !starts.contains(&matched.start()) {
                add_signal(&mut candidates, matched.as_str(), 2, 1)?;
            }
        }
    }
    let protected = evidence
        .attributed
        .into_iter()
        .map(|surface| candidate_name(&surface))
        .collect::<crate::Result<HashSet<_>>>()?;
    Ok(Phase1 {
        candidates,
        non_person,
        protected,
    })
}

/// Phase 2: Select anchors with two source categories and three votes.
/// Correlated heuristic votes are not distinct occurrences or calibrated probabilities.
fn phase2_anchors(candidates: &HashMap<String, NameSignals>) -> HashSet<String> {
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

fn phase3_learn(anchors: &HashSet<String>, scenes: &[String]) -> crate::Result<LearnedPatterns> {
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
    let mut suffixes = HashMap::<String, usize>::new();
    for anchor in anchors {
        if let Some(suffix) = name_suffix(anchor) {
            *suffixes.entry(suffix).or_default() += 1;
        }
    }
    if !anchors.is_empty() {
        let mut literals: Vec<_> = anchors.iter().map(|a| literal_pattern(a)).collect();
        literals.sort();
        let pattern = regex::RegexBuilder::new(&format!(r"(?i)\b(?:{})\b", literals.join("|")))
            .size_limit(8 * 1024 * 1024)
            .build()
            .map_err(|_| {
                crate::Error::InvalidInput("person anchor matcher exceeds limits".into())
            })?;
        for scene in scenes {
            for anchor in pattern.find_iter(scene) {
                if let Some(title) = scene[..anchor.start()].split_whitespace().next_back() {
                    let title = title
                        .trim_matches(|ch: char| !ch.is_alphabetic())
                        .to_lowercase();
                    if known_titles.contains(title.as_str()) {
                        title_prefixes.insert(title_case(&title));
                    }
                }
            }
        }
    }
    let name_suffixes = suffixes
        .into_iter()
        .filter(|(_, count)| *count >= 2)
        .map(|(s, _)| s)
        .collect();
    Ok(LearnedPatterns {
        title_prefixes,
        name_suffixes,
    })
}

fn literal_pattern(surface: &str) -> String {
    surface
        .split_whitespace()
        .map(regex::escape)
        .collect::<Vec<_>>()
        .join(r"\s+")
}
fn name_suffix(name: &str) -> Option<String> {
    (name.chars().count() >= 4).then(|| {
        name.chars()
            .rev()
            .take(3)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<String>()
            .to_lowercase()
    })
}

/// Phase 4: Extended extraction using learned patterns.
fn phase4_extend(
    candidates: &mut HashMap<String, NameSignals>,
    anchors: &HashSet<String>,
    patterns: &LearnedPatterns,
    scenes: &[String],
) -> crate::Result<HashSet<String>> {
    let mut extended = anchors.clone();

    // 4a: Promote single-signal candidates that match learned patterns
    for (name, counts) in candidates.iter() {
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
        // A listed given name, or a known honorific before a name, is
        // lexical person evidence comparable to dialogue attribution.
        if lexical_person_evidence(name) {
            confidence += 0.35;
        }
        // Capitalization alone is weak
        if counts[2] > 0 {
            confidence += 0.15;
        }

        if name_suffix(name).is_some_and(|suffix| patterns.name_suffixes.contains(&suffix)) {
            confidence += 0.15;
        }

        // Frequency contributes an uncalibrated selection vote.
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

    if !patterns.title_prefixes.is_empty() {
        let mut titles: Vec<_> = patterns
            .title_prefixes
            .iter()
            .map(|t| regex::escape(t))
            .collect();
        titles.sort();
        let pattern = regex::RegexBuilder::new(&format!(
            r"\b(?:{})\.?[ \t]+\p{{Lu}}[\p{{L}}\p{{M}}'’-]*\b",
            titles.join("|")
        ))
        .size_limit(8 * 1024 * 1024)
        .build()
        .map_err(|_| crate::Error::InvalidInput("person title matcher exceeds limits".into()))?;
        for scene in scenes {
            for hit in pattern.find_iter(scene) {
                add_signal(candidates, hit.as_str(), 3, 1)?;
            }
        }
        for (name, counts) in candidates.iter() {
            if counts[3] >= 2 {
                extended.insert(name.clone());
            }
        }
    }
    Ok(extended)
}

/// Phase 5: Structural validation.
fn phase5_validate(
    characters: &HashSet<String>,
    candidates: &HashMap<String, NameSignals>,
    non_person: &HashSet<String>,
    protected: &HashSet<String>,
) -> crate::Result<Vec<crate::continuity::entity::Entity>> {
    let mut result: Vec<(String, usize)> = Vec::new();

    for name in characters {
        let counts = candidates.get(name).copied().unwrap_or([0, 0, 0, 0]);
        let total: usize = counts.iter().sum();

        // Selection votes can describe the same observed source occurrence.
        if total < 2 {
            continue;
        }

        // A lone letter is a tagger fragment unless dialogue attribution
        // protects it; bare honorifics ("Miss", "Mr", "St.") are never names,
        // attribution included ("said Mr." stops at the period).
        if (name.chars().count() < 2 && !protected.contains(name))
            || name
                .split_whitespace()
                .all(crate::substrate::utils::is_honorific)
        {
            continue;
        }

        // Positive model/attribution categories exempt candidates from vetoes;
        // neither establishes literal personhood in figurative narrative.
        let has_person_evidence =
            counts[0] > 0 || protected.contains(name) || lexical_person_evidence(name);

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

    crate::continuity::entity::resolve_counted(
        &result,
        crate::continuity::entity::EntityType::Character,
    )
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PersonMention {
    pub character: String,
    pub surface: String,
    pub scene: usize,
    pub range: crate::manuscript::TextRange,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PersonExtractionResult {
    pub model_entities: Vec<onnx::NerSpans>,
    pub persons: Vec<(String, usize)>,
    pub mentions: Vec<PersonMention>,
    pub unassessed_mentions: usize,
    pub availability: crate::substrate::utils::AnalysisAvailability,
    pub diagnostics: Vec<String>,
    pub model_identity: Option<ModelIdentity>,
    pub model_failure: Option<ModelFailure>,
    pub model_generation: u64,
    pub ner_windows: usize,
}

/// Compatible candidate inventory. Counts are distinct located occurrences,
/// never sums of correlated model/regex/capitalization votes.
pub fn ner_extract_persons(scenes: Vec<String>) -> Vec<(String, usize)> {
    ner_extract_persons_checked(&scenes)
        .map(|r| r.persons)
        .unwrap_or_default()
}

pub fn ner_extract_persons_checked(scenes: &[String]) -> crate::Result<PersonExtractionResult> {
    ner_extract_persons_with_backend_checked(onnx::backend(), scenes)
}

pub fn ner_extract_persons_with_backend_checked(
    backend: &dyn onnx::InferenceBackend,
    scenes: &[String],
) -> crate::Result<PersonExtractionResult> {
    validate_person_texts(scenes)?;
    let generation = onnx::model_generation();
    let texts: Vec<_> = scenes.iter().map(String::as_str).collect();
    let (entities, identity, failure, windows) =
        match ner_extract_entities_batch_with_backend_checked(backend, &texts) {
            Ok(result) => (result.entities, result.model_identity, None, result.windows),
            Err(error) => (vec![Vec::new(); scenes.len()], None, Some(error), 0),
        };
    let Phase1 {
        mut candidates,
        non_person,
        protected,
    } = phase1_collect(scenes, &entities)?;
    let anchors = phase2_anchors(&candidates);
    let patterns = phase3_learn(&anchors, scenes)?;
    let mut extended = phase4_extend(&mut candidates, &anchors, &patterns, scenes)?;
    extended.extend(protected.iter().cloned());
    extended.extend(
        candidates
            .iter()
            .filter(|(_, counts)| counts[0] >= 2)
            .map(|(name, _)| name.clone()),
    );
    let selected = phase5_validate(&extended, &candidates, &non_person, &protected)?;
    let (persons, mentions, unassessed) = observe_persons(scenes, &selected)?;
    if onnx::model_generation() != generation {
        return Err(crate::Error::AnalysisFailed(
            "model_replaced_during_person_extraction".into(),
        ));
    }
    let mut diagnostics = vec!["heuristic_candidate_inventory_not_verified_character_identity; source_votes_are_uncalibrated_and_correlated".into(),
        "counts_are_unique_longest_literal_occurrences; original_byte_and_utf16_spans_preserved".into()];
    if let Some(error) = &failure {
        diagnostics.push(format!("model_fallback:{}", error.code));
    }
    if unassessed > 0 {
        diagnostics.push("ambiguous_literal_identity_occurrences_unassessed".into());
    }
    Ok(PersonExtractionResult {
        model_entities: entities,
        persons,
        mentions,
        unassessed_mentions: unassessed,
        availability: crate::substrate::utils::AnalysisAvailability::Partial,
        diagnostics,
        model_identity: identity,
        model_failure: failure,
        model_generation: generation,
        ner_windows: windows,
    })
}

fn validate_person_texts(scenes: &[String]) -> crate::Result<()> {
    if scenes.len() > MAX_NER_TEXTS {
        return Err(crate::Error::InvalidInput(
            "person scene count exceeds limit".into(),
        ));
    }
    let mut bytes = 0usize;
    let mut tokens = 0usize;
    let mut markers = 0usize;
    for text in scenes {
        bytes += text.len();
        if text.len() > MAX_NER_TEXT_BYTES || bytes > MAX_NER_TOTAL_BYTES {
            return Err(crate::Error::InvalidInput(
                "person source byte budget exceeded".into(),
            ));
        }
        tokens += text.split_whitespace().take(MAX_PERSON_TOKENS + 1).count();
        markers += text
            .chars()
            .filter(|ch| matches!(ch, '.' | '!' | '?' | '\u{2026}'))
            .take(MAX_PERSON_MARKERS + 1)
            .count();
        if tokens > MAX_PERSON_TOKENS || markers > MAX_PERSON_MARKERS {
            return Err(crate::Error::InvalidInput(
                "person token/marker budget exceeded".into(),
            ));
        }
    }
    Ok(())
}

type ObservedPersons = (Vec<(String, usize)>, Vec<PersonMention>, usize);
fn observe_persons(
    scenes: &[String],
    entities: &[crate::continuity::entity::Entity],
) -> crate::Result<ObservedPersons> {
    if entities.is_empty() {
        return Ok((Vec::new(), Vec::new(), 0));
    }
    let mut aliases = Vec::new();
    for (index, entity) in entities.iter().enumerate() {
        for alias in std::iter::once(&entity.canonical).chain(&entity.aliases) {
            aliases.push((index, literal_pattern(alias)));
        }
    }
    let identities =
        regex::RegexSetBuilder::new(aliases.iter().map(|(_, alias)| format!("(?i)^{alias}$")))
            .size_limit(8 * 1024 * 1024)
            .build()
            .map_err(|_| {
                crate::Error::InvalidInput("person identity matcher exceeds limits".into())
            })?;
    let mut literals: Vec<_> = aliases.iter().map(|(_, alias)| alias.as_str()).collect();
    literals.sort_unstable_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    literals.dedup();
    let words = regex::RegexBuilder::new(&format!(r"(?i)\b(?:{})\b", literals.join("|")))
        .size_limit(8 * 1024 * 1024)
        .build()
        .map_err(|_| crate::Error::InvalidInput("person literal matcher exceeds limits".into()))?;
    let mut observations = Vec::new();
    let mut unassessed = 0;
    let mut surfaces = vec![BTreeMap::<String, usize>::new(); entities.len()];
    for (scene, text) in scenes.iter().enumerate() {
        let mut cursor = 0;
        let mut utf16_cursor = 0u32;
        for hit in words.find_iter(text) {
            if observations.len() + unassessed >= MAX_NER_ENTITIES {
                return Err(crate::Error::InvalidInput(
                    "person observed mention budget exceeded".into(),
                ));
            }
            let candidates: BTreeSet<_> = identities
                .matches(hit.as_str())
                .iter()
                .map(|i| aliases[i].0)
                .collect();
            let utf16_start =
                utf16_cursor + text[cursor..hit.start()].encode_utf16().count() as u32;
            let utf16_end = utf16_start + hit.as_str().encode_utf16().count() as u32;
            cursor = hit.end();
            utf16_cursor = utf16_end;
            if candidates.len() != 1 {
                unassessed += 1;
                continue;
            }
            let index = *candidates.first().expect("one literal identity");
            let surface = hit
                .as_str()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            *surfaces[index].entry(surface).or_default() += 1;
            observations.push((
                index,
                PersonMention {
                    character: String::new(),
                    surface: hit.as_str().into(),
                    scene,
                    range: crate::manuscript::TextRange {
                        byte_start: hit.start(),
                        byte_end: hit.end(),
                        utf16_start,
                        utf16_end,
                    },
                },
            ));
        }
    }
    let mut persons = Vec::new();
    let mut canonical = Vec::with_capacity(entities.len());
    for (entity, surfaces) in entities.iter().zip(surfaces) {
        let name = surfaces
            .iter()
            .filter(|(surface, _)| surface.to_lowercase() == entity.canonical.to_lowercase())
            .max_by(|(a, an), (b, bn)| an.cmp(bn).then_with(|| b.cmp(a)))
            .map_or_else(|| entity.canonical.clone(), |(s, _)| s.clone());
        let count: usize = surfaces.values().sum();
        if count > 0 {
            persons.push((name.clone(), count));
        }
        canonical.push(name);
    }
    let mentions = observations
        .into_iter()
        .map(|(index, mut mention)| {
            mention.character = canonical[index].clone();
            mention
        })
        .collect();
    persons.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Ok((persons, mentions, unassessed))
}

/// Whether the surface reads as a person by its words alone: a known
/// honorific before a further token, or a listed given name as the first name
/// token. "St." is excluded because it mostly prefixes places.
fn lexical_person_evidence(name: &str) -> bool {
    let mut tokens = name.split_whitespace().peekable();
    let mut titled = false;
    while let Some(token) = tokens.peek() {
        let bare = token.trim_end_matches('.');
        if crate::substrate::utils::is_honorific(bare)
            && !matches!(bare.to_lowercase().as_str(), "st" | "saint")
        {
            titled = true;
            tokens.next();
        } else {
            break;
        }
    }
    tokens.next().is_some_and(|first| {
        let first = first.trim_end_matches("'s").trim_end_matches("’s");
        titled || crate::substrate::given_names::given_name_gender(first).is_some()
    })
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
    fn checked_batches_discard_all_output_if_generation_changes() {
        struct Names;
        impl onnx::InferenceBackend for Names {
            fn embed(&self, texts: &[&str]) -> Vec<Vec<f64>> {
                vec![Vec::new(); texts.len()]
            }
            fn check_model(&self, _: ModelKind) -> Result<(), ModelFailure> {
                Ok(())
            }
            fn tag_batch_checked(
                &self,
                texts: &[&str],
            ) -> Result<Vec<onnx::NerSpans>, ModelFailure> {
                Ok(self.tag_batch(texts))
            }
            fn tag(&self, text: &str) -> onnx::NerSpans {
                vec![(text.to_string(), "PER".into(), 0, text.len())]
            }
        }
        let calls = std::cell::Cell::new(0u64);
        let generation = || {
            let value = calls.get();
            calls.set(value + 1);
            value
        };
        assert_eq!(
            extract_checked(&Names, &["Alice"], generation)
                .unwrap_err()
                .code,
            "model_replaced_during_ner"
        );
        let valid = extract_checked(&Names, &["Alice"], || 7).unwrap();
        assert_eq!(valid.model_generation, 7);
        assert_eq!(valid.entities[0][0].0, "Alice");
    }

    #[test]
    fn test_title_case() {
        assert_eq!(title_case("john doe"), "John Doe");
        assert_eq!(title_case("ALICE"), "Alice");
        assert_eq!(title_case(""), "");
    }

    #[test]
    fn test_phase2_anchors() {
        let mut cands = HashMap::new();
        cands.insert("Alice".to_string(), [5, 3, 2, 0]); // all 3 signals
        cands.insert("Bob".to_string(), [2, 0, 1, 0]); // 2 signals, 3 total
        cands.insert("Noise".to_string(), [0, 0, 1, 0]); // 1 signal
        let anchors = phase2_anchors(&cands);
        assert!(anchors.contains("Alice"));
        assert!(anchors.contains("Bob"));
        assert!(!anchors.contains("Noise"));
    }

    #[test]
    fn suffixes_count_unique_anchors_and_unicode_characters() {
        let one = HashSet::from(["Alice".into()]);
        let scenes = vec!["Alice passed.".into(), "Alice rested.".into()];
        assert!(
            phase3_learn(&one, &scenes)
                .unwrap()
                .name_suffixes
                .is_empty()
        );
        let two = HashSet::from(["Élóra".into(), "Flóra".into()]);
        assert_eq!(
            phase3_learn(&two, &[]).unwrap().name_suffixes,
            HashSet::from(["óra".into()])
        );
        assert_eq!(name_suffix("Aé你"), None);
        assert_eq!(name_suffix("ABé你").as_deref(), Some("bé你"));
    }

    #[test]
    fn learned_titles_use_later_occurrences_and_supply_real_selection_evidence() {
        let anchors = HashSet::from(["Marcus".into()]);
        let scenes = vec![
            "Marcus waited. Lord Marcus left. Lord Élodie arrived. Lord Élodie rested.".into(),
        ];
        let patterns = phase3_learn(&anchors, &scenes).unwrap();
        assert!(patterns.title_prefixes.contains("Lord"));
        let mut candidates = HashMap::new();
        let extended = phase4_extend(&mut candidates, &anchors, &patterns, &scenes).unwrap();
        assert_eq!(candidates["Lord Élodie"][3], 2);
        let entities =
            phase5_validate(&extended, &candidates, &HashSet::new(), &HashSet::new()).unwrap();
        let (persons, mentions, _) = observe_persons(&scenes, &entities).unwrap();
        assert_eq!(persons, [("Lord Élodie".into(), 2)]);
        assert_eq!(mentions.len(), 2);
    }

    #[test]
    fn test_phase5_filters_short() {
        let mut chars = HashSet::new();
        chars.insert("Al".to_string());
        chars.insert("Alice".to_string());
        let mut cands = HashMap::new();
        cands.insert("Al".to_string(), [1, 0, 0, 0]);
        cands.insert("Alice".to_string(), [3, 2, 1, 0]);
        let result = phase5_validate(&chars, &cands, &HashSet::new(), &HashSet::new()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].canonical, "Alice");
    }

    #[test]
    fn phase5_rejects_tagger_fragments_but_keeps_titled_names_and_protected_initials() {
        let names = ["M", "Miss", "St", "Miss Eldridge", "X"];
        let chars: HashSet<_> = names.iter().map(|name| name.to_string()).collect();
        let cands: HashMap<_, _> = names
            .iter()
            .map(|name| (name.to_string(), [4, 0, 0, 0]))
            .collect();
        let protected = HashSet::from(["X".to_string()]);
        let result = phase5_validate(&chars, &cands, &HashSet::new(), &protected).unwrap();
        let kept: HashSet<_> = result
            .iter()
            .map(|entity| entity.canonical.as_str())
            .collect();
        assert_eq!(kept, HashSet::from(["Miss Eldridge", "X"]));
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
        cands.insert("Paris".to_string(), [0, 2, 3, 0]); // place, no PER
        cands.insert("City".to_string(), [0, 2, 2, 0]); // common word, no PER
        cands.insert("Marcus".to_string(), [0, 0, 4, 0]); // real name, cap-only
        cands.insert("Just".to_string(), [3, 0, 2, 0]); // common word but PER-tagged
        let non_person = HashSet::from(["Paris".to_string()]);
        let result = phase5_validate(&chars, &cands, &non_person, &HashSet::new()).unwrap();
        let names: HashSet<&str> = result
            .iter()
            .map(|entity| entity.canonical.as_str())
            .collect();
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
        cands.insert("Paris".to_string(), [0, 4, 3, 0]); // no PER signal
        let non_person = HashSet::from(["Paris".to_string()]); // NER called it a place
        let protected = HashSet::from(["Paris".to_string()]); // but "Paris said ..."
        let result = phase5_validate(&chars, &cands, &non_person, &protected).unwrap();
        let names: HashSet<&str> = result
            .iter()
            .map(|entity| entity.canonical.as_str())
            .collect();
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
    fn windows_preserve_unicode_whitespace_boundaries_at_the_limit() {
        for whitespace in ['\u{0085}', '\u{00a0}', '\u{2003}', '\u{3000}'] {
            for end in [1493usize, 1499, 1500, 1501] {
                let text = format!(
                    "{}{}{}",
                    "a".repeat(end - whitespace.len_utf8()),
                    whitespace,
                    "b".repeat(20)
                );
                let spans = ner_windows(&text, NER_WINDOW_BYTES);
                assert_eq!(spans[0].0, 0);
                assert_eq!(spans.last().unwrap().1, text.len());
                for (i, &(start, end)) in spans.iter().enumerate() {
                    assert!(end > start && end - start <= NER_WINDOW_BYTES);
                    assert!(text.is_char_boundary(start) && text.is_char_boundary(end));
                    if i > 0 {
                        assert_eq!(spans[i - 1].1, start);
                    }
                }
            }
        }
    }

    #[test]
    fn person_validation_preserves_conflicting_full_names_and_counts() {
        let names = ["John Smith", "Smith", "Mary Smith"];
        let chars = names.iter().map(|name| name.to_string()).collect();
        let candidates = names
            .iter()
            .map(|name| (name.to_string(), [3, 0, 0, 0]))
            .collect();
        let result: Vec<_> = phase5_validate(&chars, &candidates, &HashSet::new(), &HashSet::new())
            .unwrap()
            .into_iter()
            .map(|e| (e.canonical, e.mention_count))
            .collect();
        assert_eq!(
            result,
            vec![
                ("John Smith".into(), 3),
                ("Mary Smith".into(), 3),
                ("Smith".into(), 3)
            ]
        );
    }

    #[test]
    fn test_graceful_no_model() {
        let result = ner_extract_persons(vec!["John went to Paris.".to_string()]);
        let _ = result;
    }
}
