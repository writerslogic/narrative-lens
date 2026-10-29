use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::OnceLock;

use super::author_knowledge::AuthorSource;
use crate::{
    Error, Result,
    craft::lexicon::is_word_char,
    manuscript::{
        Evidence, Manuscript, TextRange, author_source::validate_sources, config::validate_id,
    },
    substrate::utils::AnalysisAvailability,
};
use serde::{Deserialize, Serialize};

pub const MAX_HISTORICAL_ENTRIES: usize = 10_000;
pub const MAX_HISTORICAL_USAGES: usize = 10_000;
pub const MAX_HISTORICAL_CANDIDATES: usize = 10_000;
pub const MAX_HISTORICAL_TOKENS: usize = 250_000;
pub const MAX_HISTORICAL_TERM_BYTES: usize = 128;
pub const MAX_HISTORICAL_TERM_TOKENS: usize = 32;
pub const MAX_HISTORICAL_METADATA_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_HISTORICAL_MATCH_BYTES: usize = 4096;

/// A legacy lexicon date-review candidate; historical dates/senses are unverified.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnachronismItem {
    pub word: String,
    pub introduced_year: i32,
    pub target_year: i32,
    pub scene_id: usize,
}

const GRACE_PERIOD_YEARS: i32 = 20;

fn milestones() -> &'static HashMap<&'static str, i32> {
    static M: OnceLock<HashMap<&str, i32>> = OnceLock::new();
    M.get_or_init(|| {
        let mut m = HashMap::new();
        // Communication & media
        m.insert("internet", 1991);
        m.insert("email", 1972);
        m.insert("texting", 1992);
        m.insert("blog", 1997);
        m.insert("podcast", 2004);
        m.insert("selfie", 2002);
        m.insert("hashtag", 2007);
        m.insert("viral", 1990);
        m.insert("streaming", 1995);
        m.insert("wifi", 1997);
        m.insert("bluetooth", 1998);
        m.insert("emoji", 1999);
        m.insert("meme", 1976);
        m.insert("tweet", 2006);
        m.insert("google", 1998);
        // Technology
        m.insert("computer", 1940);
        m.insert("smartphone", 2007);
        m.insert("laptop", 1983);
        m.insert("software", 1958);
        m.insert("television", 1927);
        m.insert("radio", 1895);
        m.insert("telephone", 1876);
        m.insert("telegraph", 1837);
        m.insert("microchip", 1958);
        m.insert("transistor", 1947);
        m.insert("laser", 1960);
        m.insert("radar", 1935);
        m.insert("satellite", 1957);
        m.insert("robot", 1920);
        m.insert("algorithm", 1950);
        m.insert("database", 1960);
        m.insert("pixel", 1965);
        m.insert("digital", 1942);
        m.insert("analog", 1940);
        m.insert("cybernetic", 1948);
        // Electronics
        m.insert("semiconductor", 1950);
        m.insert("microprocessor", 1971);
        m.insert("silicon", 1824);
        m.insert("circuit", 1899);
        m.insert("diode", 1919);
        m.insert("capacitor", 1926);
        m.insert("resistor", 1909);
        m.insert("oscilloscope", 1897);
        m.insert("solenoid", 1823);
        // Transport
        m.insert("airplane", 1903);
        m.insert("helicopter", 1939);
        m.insert("automobile", 1886);
        m.insert("motorcycle", 1885);
        m.insert("subway", 1863);
        m.insert("escalator", 1897);
        m.insert("rocket", 1926);
        m.insert("spaceship", 1929);
        m.insert("jeep", 1940);
        m.insert("tank", 1916);
        m.insert("submarine", 1897);
        // Science & medicine
        m.insert("antibiotic", 1928);
        m.insert("penicillin", 1928);
        m.insert("vaccine", 1796);
        m.insert("x-ray", 1895);
        m.insert("anesthesia", 1846);
        m.insert("aspirin", 1899);
        m.insert("insulin", 1921);
        m.insert("dna", 1953);
        m.insert("gene", 1909);
        m.insert("vitamin", 1912);
        m.insert("calorie", 1824);
        m.insert("electron", 1897);
        m.insert("atom", 1803);
        m.insert("neutron", 1932);
        m.insert("radioactive", 1898);
        m.insert("plastic", 1907);
        m.insert("nylon", 1935);
        m.insert("dynamite", 1867);
        m.insert("biodiversity", 1985);
        // Medical instruments & drugs
        m.insert("stethoscope", 1816);
        m.insert("thermometer", 1714);
        m.insert("syringe", 1853);
        m.insert("morphine", 1804);
        m.insert("chloroform", 1831);
        m.insert("defibrillator", 1947);
        m.insert("ultrasound", 1956);
        m.insert("ibuprofen", 1961);
        m.insert("dialysis", 1943);
        m.insert("pacemaker", 1958);
        m.insert("prosthetic", 1912);
        m.insert("antiseptic", 1867);
        m.insert("bandaid", 1920);
        // Weapons
        m.insert("gatling", 1862);
        m.insert("torpedo", 1800);
        m.insert("landmine", 1277);
        m.insert("grenade", 1500);
        m.insert("napalm", 1942);
        m.insert("missile", 1611);
        m.insert("bayonet", 1670);
        m.insert("revolver", 1836);
        m.insert("carbine", 1605);
        m.insert("howitzer", 1695);
        m.insert("mortar", 1453);
        m.insert("musket", 1499);
        m.insert("flintlock", 1610);
        m.insert("matchlock", 1411);
        m.insert("blunderbuss", 1654);
        m.insert("shrapnel", 1784);
        // Clothing
        m.insert("jeans", 1873);
        m.insert("sneakers", 1895);
        m.insert("bra", 1914);
        m.insert("bikini", 1946);
        m.insert("hoodie", 1930);
        m.insert("tuxedo", 1888);
        m.insert("raincoat", 1830);
        m.insert("cardigan", 1868);
        m.insert("khaki", 1848);
        m.insert("parka", 1780);
        // Music
        m.insert("phonograph", 1877);
        m.insert("microphone", 1877);
        m.insert("amplifier", 1912);
        m.insert("synthesizer", 1955);
        m.insert("gramophone", 1887);
        m.insert("jukebox", 1927);
        m.insert("turntable", 1925);
        m.insert("headphones", 1910);
        m.insert("loudspeaker", 1898);
        m.insert("saxophone", 1846);
        m.insert("harmonica", 1821);
        m.insert("accordion", 1822);
        // Sports
        m.insert("basketball", 1891);
        m.insert("volleyball", 1895);
        m.insert("baseball", 1845);
        m.insert("tennis", 1874);
        m.insert("badminton", 1873);
        m.insert("hockey", 1875);
        m.insert("rugby", 1823);
        m.insert("cricket", 1598);
        // Food & drink
        m.insert("hamburger", 1884);
        m.insert("chocolate-chip", 1938);
        m.insert("sushi", 1820);
        m.insert("ketchup", 1812);
        m.insert("cornflakes", 1894);
        m.insert("pretzel", 1510);
        m.insert("croissant", 1838);
        m.insert("espresso", 1884);
        m.insert("cappuccino", 1930);
        m.insert("brunch", 1895);
        m.insert("cocktail", 1806);
        // Social & political
        m.insert("feminism", 1837);
        m.insert("socialism", 1832);
        m.insert("capitalism", 1850);
        m.insert("communism", 1840);
        m.insert("environmentalist", 1960);
        m.insert("genocide", 1944);
        m.insert("propaganda", 1622);
        m.insert("boycott", 1880);
        m.insert("sabotage", 1910);
        m.insert("fascism", 1919);
        m.insert("racism", 1902);
        m.insert("totalitarian", 1926);
        m.insert("anarchism", 1642);
        m.insert("imperialism", 1826);
        m.insert("nationalism", 1798);
        // Everyday items
        m.insert("zipper", 1917);
        m.insert("refrigerator", 1913);
        m.insert("microwave", 1947);
        m.insert("dishwasher", 1886);
        m.insert("toaster", 1909);
        m.insert("blender", 1922);
        m.insert("vacuum", 1901);
        m.insert("lightbulb", 1879);
        m.insert("battery", 1800);
        m.insert("photograph", 1839);
        m.insert("camera", 1816);
        m.insert("wristwatch", 1868);
        m.insert("thermos", 1904);
        m.insert("ballpoint", 1938);
        m.insert("stapler", 1877);
        m.insert("paperclip", 1867);
        m.insert("eraser", 1770);
        // Textiles & fashion
        m.insert("polyester", 1941);
        m.insert("spandex", 1958);
        m.insert("velcro", 1941);
        m.insert("rayon", 1924);
        m.insert("lycra", 1958);
        // Food & drink (additional)
        m.insert("pasteurize", 1862);
        m.insert("margarine", 1869);
        m.insert("coca-cola", 1886);
        // Language / slang
        m.insert("groovy", 1937);
        m.insert("dude", 1883);
        // Misc
        m.insert("automation", 1946);
        m.insert("feedback", 1920);
        m.insert("logistics", 1846);
        m.insert("ok", 1839);
        m.insert("pipeline", 1900);
        m.insert("deadline", 1864);
        m.insert("freelance", 1820);
        m.insert("bureaucracy", 1818);
        m.insert("stereotype", 1798);
        m.insert("silhouette", 1783);
        m.insert("panorama", 1791);
        m.insert("electricity", 1646);
        m.insert("barometer", 1666);
        m.insert("pendulum", 1643);
        m.insert("telescope", 1611);
        m.insert("microscope", 1625);
        m.insert("encyclopedia", 1541);
        m
    })
}

fn multi_word_milestones() -> &'static Vec<(&'static str, i32)> {
    static M: OnceLock<Vec<(&str, i32)>> = OnceLock::new();
    M.get_or_init(|| {
        vec![
            ("cell phone", 1983),
            ("social media", 2004),
            ("world wide web", 1991),
            ("virtual reality", 1987),
            ("artificial intelligence", 1956),
            ("credit card", 1950),
            ("fast food", 1951),
            ("global warming", 1975),
            ("assembly line", 1913),
            ("birth control", 1914),
            ("machine learning", 1959),
            ("nuclear power", 1945),
            ("space shuttle", 1981),
            ("t-shirt", 1904),
            ("hot-dog", 1867),
            ("mp3 player", 1998),
            ("blood transfusion", 1818),
            ("contact lens", 1887),
            ("dry cleaning", 1855),
            ("ice cream", 1744),
            ("chewing gum", 1848),
            ("barbed wire", 1873),
            ("sewing machine", 1846),
            ("typewriter", 1868),
            ("washing machine", 1851),
            ("printing press", 1440),
            ("steam engine", 1712),
            ("internal combustion", 1860),
            ("stock market", 1611),
            ("central bank", 1668),
            ("paper money", 1690),
        ]
    })
}

/// Concepts that are anachronistic in medieval settings (target_year < 1500),
/// even if the specific word existed earlier in other contexts.
fn medieval_anachronisms() -> &'static Vec<(&'static str, &'static str)> {
    static M: OnceLock<Vec<(&str, &str)>> = OnceLock::new();
    M.get_or_init(|| {
        vec![
            ("democracy", "modern democratic concept"),
            ("evolution", "evolutionary theory"),
            ("psychology", "psychological science"),
            ("genetics", "genetic science"),
            ("bureaucracy", "bureaucratic systems"),
            ("constitution", "constitutional government"),
            ("parliament", "parliamentary government"),
            ("republic", "republican government"),
            ("capitalism", "capitalist economy"),
            ("socialism", "socialist ideology"),
            ("communism", "communist ideology"),
            ("nationalism", "nationalist movement"),
            ("revolution", "political revolution concept"),
            ("ideology", "ideological framework"),
            ("civilization", "civilizational concept"),
            ("education", "formal education system"),
            ("university", "university institution"),
            ("newspaper", "printed newspaper"),
            ("patent", "patent system"),
            ("copyright", "copyright law"),
            ("insurance", "insurance system"),
            ("factory", "industrial factory"),
            ("corporation", "corporate entity"),
            ("investment", "financial investment"),
            ("inflation", "economic inflation"),
        ]
    })
}

/// Slang terms that are anachronistic if used before their era.
fn slang_milestones() -> &'static Vec<(&'static str, i32)> {
    static M: OnceLock<Vec<(&str, i32)>> = OnceLock::new();
    M.get_or_init(|| {
        vec![
            ("cool", 1933),
            ("okay", 1839),
            ("awesome", 1980),
            ("bummer", 1966),
            ("chill", 1979),
            ("vibe", 1967),
            ("gig", 1926),
            ("hip", 1904),
            ("jazz", 1912),
            ("punk", 1596),
            ("geek", 1916),
            ("nerd", 1951),
            ("binge", 1854),
            ("hype", 1931),
            ("mob", 1688),
            ("radar", 1935),
            ("snafu", 1941),
        ]
    })
}

fn is_anachronistic(introduced_year: i32, target_year: i32) -> bool {
    i64::from(introduced_year) > i64::from(target_year) + i64::from(GRACE_PERIOD_YEARS)
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnachronismAnalysisResult {
    pub items: Vec<AnachronismItem>,
    pub availability: AnalysisAvailability,
    pub diagnostics: Vec<String>,
    pub abstained_terms: Vec<String>,
}

pub fn analyze_scene(tokens: &[String], target_year: i32, scene_id: usize) -> Vec<AnachronismItem> {
    analyze_scene_checked(tokens, target_year, scene_id)
        .map(|r| r.items)
        .unwrap_or_default()
}

pub fn analyze_scene_checked(
    tokens: &[String],
    target_year: i32,
    scene_id: usize,
) -> Result<AnachronismAnalysisResult> {
    if tokens.len() > MAX_HISTORICAL_TOKENS || scene_id >= 10_000 {
        return Err(Error::InvalidInput(
            "historical token count or scene index exceeds limit".into(),
        ));
    }
    let mut bytes = 0usize;
    for token in tokens {
        bytes += token.len();
        if token.len() > 4096 || bytes > 16 * 1024 * 1024 || token.contains('\0') {
            return Err(Error::InvalidInput(
                "historical token data exceeds limits".into(),
            ));
        }
    }
    let mut result = AnachronismAnalysisResult {
        items: Vec::new(), availability: AnalysisAvailability::Partial,
        diagnostics: vec!["builtin_dates_unverified; lexical_review_candidates_not_historical_incompatibility_proof".into(),
            "polysemous_slang_and_unsourced_period_concepts_abstain; use_referenced_catalogue_and_explicit_sense_annotations".into()],
        abstained_terms: Vec::new(),
    };
    if target_year == 0 {
        return Ok(result);
    }
    let lower: Vec<_> = tokens.iter().map(|t| t.to_lowercase()).collect();
    let mut ambiguous: BTreeSet<&str> = [
        "computer",
        "viral",
        "streaming",
        "tweet",
        "tablet",
        "cloud",
        "battery",
        "coach",
        "train",
        "tank",
        "vacuum",
        "atom",
        "camera",
        "pipeline",
        "feedback",
        "revolution",
        "constitution",
        "republic",
        "education",
        "university",
        "insurance",
    ]
    .into_iter()
    .collect();
    ambiguous.extend(slang_milestones().iter().map(|(term, _)| *term));
    ambiguous.extend(medieval_anachronisms().iter().map(|(term, _)| *term));
    let mut abstained = BTreeSet::new();
    for &(term, introduced_year) in multi_word_milestones() {
        let words: Vec<_> = term.split_whitespace().collect();
        if lower
            .windows(words.len())
            .any(|window| window.iter().map(String::as_str).eq(words.iter().copied()))
            && is_anachronistic(introduced_year, target_year)
        {
            result.items.push(AnachronismItem {
                word: term.into(),
                introduced_year,
                target_year,
                scene_id,
            });
        }
    }
    for (token, lower) in tokens.iter().zip(&lower) {
        if let Some(&introduced_year) = milestones().get(lower.as_str())
            && is_anachronistic(introduced_year, target_year)
        {
            if ambiguous.contains(lower.as_str()) {
                abstained.insert(lower.clone());
            } else {
                result.items.push(AnachronismItem {
                    word: token.clone(),
                    introduced_year,
                    target_year,
                    scene_id,
                });
                if result.items.len() > MAX_HISTORICAL_CANDIDATES {
                    return Err(Error::InvalidInput(
                        "historical candidate count exceeds limit".into(),
                    ));
                }
            }
        }
    }
    for &(term, year) in slang_milestones() {
        if is_anachronistic(year, target_year) && lower.iter().any(|token| token == term) {
            abstained.insert(term.into());
        }
    }
    if target_year < 1500 {
        for &(term, _) in medieval_anachronisms() {
            if lower.iter().any(|token| token == term) {
                abstained.insert(term.into());
            }
        }
    }
    if result.items.len() > MAX_HISTORICAL_CANDIDATES {
        return Err(Error::InvalidInput(
            "historical candidate count exceeds limit".into(),
        ));
    }
    result.abstained_terms = abstained.into_iter().collect();
    Ok(result)
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoricalBasis {
    FirstAttestation,
    Availability,
    AuthorConstraint,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct HistoricalEntry {
    pub id: String,
    pub term: String,
    pub earliest_year: i32,
    pub basis: HistoricalBasis,
    pub reference: String,
    pub context: String,
    pub sense_ambiguous: bool,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct HistoricalUsage {
    pub id: String,
    pub entry_id: String,
    pub evidence: AuthorSource,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HistoricalCandidate {
    pub entry_id: String,
    pub scene_id: String,
    pub range: TextRange,
    pub quote: String,
    /// Integer-date comparison only, never proof of impossibility. Unassessed
    /// when the catalogue requires sense confirmation and none was supplied.
    pub date_conflict: Option<bool>,
    pub author_usage_id: Option<String>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HistoricalAnalysisResult {
    pub document_id: String,
    pub target_year: i32,
    pub tolerance_years: u32,
    pub catalogue: Vec<HistoricalEntry>,
    pub candidates: Vec<HistoricalCandidate>,
    pub availability: AnalysisAvailability,
    pub diagnostics: Vec<String>,
}

/// Lexical tokens with byte offsets: each maximal run of word characters, and
/// each other non-whitespace character on its own, so "O.K." is `O . K .`
/// whether or not spaces separate the marks.
fn historical_tokens(text: &str) -> impl Iterator<Item = (usize, &str)> {
    let mut chars = text.char_indices().peekable();
    std::iter::from_fn(move || {
        let (start, c) = chars.by_ref().find(|&(_, c)| !c.is_whitespace())?;
        let mut end = start + c.len_utf8();
        if is_word_char(c) {
            while let Some(&(i, c)) = chars.peek() {
                if !is_word_char(c) {
                    break;
                }
                end = i + c.len_utf8();
                chars.next();
            }
        }
        Some((start, &text[start..end]))
    })
}

fn term_key(term: &str, max_bytes: usize) -> Result<Vec<(bool, String)>> {
    if term.trim().is_empty()
        || term.len() > max_bytes
        || term
            .chars()
            .any(|ch| ch.is_control() && !ch.is_whitespace())
        || !term.chars().any(char::is_alphanumeric)
    {
        return Err(Error::InvalidInput(
            "historical term requires nonempty bounded lexical text".into(),
        ));
    }
    let mut end = 0;
    let mut tokens = Vec::new();
    for (start, token) in historical_tokens(term) {
        tokens.push((
            !tokens.is_empty() && !term[end..start].is_empty(),
            token.to_lowercase(),
        ));
        end = start + token.len();
        if tokens.len() > MAX_HISTORICAL_TERM_TOKENS {
            return Err(Error::InvalidInput(
                "historical term token count exceeds limit".into(),
            ));
        }
    }
    Ok(tokens)
}

#[derive(Default)]
struct TermNode {
    children: BTreeMap<(bool, String), usize>,
    entries: Vec<usize>,
}

fn charge(total: &mut usize, bytes: usize) -> Result<()> {
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| Error::InvalidInput("historical metadata size overflow".into()))?;
    if *total > MAX_HISTORICAL_METADATA_BYTES {
        return Err(Error::InvalidInput(
            "historical metadata exceeds byte limit".into(),
        ));
    }
    Ok(())
}

/// Matches a referenced caller catalogue. Dates are supplied integer constraints;
/// first attestation does not establish first use. Quotes confirm annotation
/// location and declared sense, not entailment, existence, or story chronology.
pub fn analyze_historical_catalogue(
    manuscript: &Manuscript,
    catalogue: &[HistoricalEntry],
    usages: &[HistoricalUsage],
    target_year: i32,
    tolerance_years: u32,
) -> Result<HistoricalAnalysisResult> {
    if catalogue.len() > MAX_HISTORICAL_ENTRIES || usages.len() > MAX_HISTORICAL_USAGES {
        return Err(Error::InvalidInput(
            "historical catalogue/usage count exceeds limit".into(),
        ));
    }
    manuscript.validate()?;
    let mut ids = BTreeMap::new();
    let mut keys = Vec::with_capacity(catalogue.len());
    let mut bytes = 0;
    let mut nodes = vec![TermNode::default()];
    for (index, entry) in catalogue.iter().enumerate() {
        validate_id(&entry.id)?;
        if ids.insert(entry.id.as_str(), index).is_some()
            || entry.reference.trim().is_empty()
            || entry.reference.len() > 4096
            || entry.context.trim().is_empty()
            || entry.context.len() > 1024
            || entry.reference.contains('\0')
            || entry.context.contains('\0')
        {
            return Err(Error::InvalidInput(
                "historical entry requires unique ID, reference and context within limits".into(),
            ));
        }
        let key = term_key(&entry.term, MAX_HISTORICAL_TERM_BYTES)?;
        charge(
            &mut bytes,
            entry.id.len() + entry.term.len() + entry.reference.len() + entry.context.len(),
        )?;
        let mut current = 0;
        for part in &key {
            current = match nodes[current].children.get(part) {
                Some(&next) => next,
                None => {
                    let next = nodes.len();
                    nodes.push(TermNode::default());
                    nodes[current].children.insert(part.clone(), next);
                    next
                }
            };
        }
        nodes[current].entries.push(index);
        keys.push(key);
    }
    let mut usage_ids = BTreeSet::new();
    for usage in usages {
        validate_id(&usage.id)?;
        validate_id(&usage.entry_id)?;
        let index = ids.get(usage.entry_id.as_str()).ok_or_else(|| {
            Error::InvalidInput("historical usage references unknown entry".into())
        })?;
        if !usage_ids.insert(usage.id.as_str())
            || usage.evidence.quote.trim() != usage.evidence.quote
            || term_key(&usage.evidence.quote, MAX_HISTORICAL_MATCH_BYTES)? != keys[*index]
        {
            return Err(Error::InvalidInput(
                "historical usage requires unique ID and exact term quotation".into(),
            ));
        }
        charge(
            &mut bytes,
            usage.id.len()
                + usage.entry_id.len()
                + usage.evidence.scene_id.len()
                + usage.evidence.quote.len(),
        )?;
    }
    let sources = validate_sources(
        manuscript,
        &usages.iter().map(|u| &u.evidence).collect::<Vec<_>>(),
    )?;
    let mut confirmed = BTreeMap::new();
    for (usage, evidence) in usages.iter().zip(sources) {
        let Evidence {
            scene_id, range, ..
        } = evidence;
        if confirmed
            .insert(
                (
                    usage.entry_id.as_str(),
                    scene_id,
                    range.byte_start,
                    range.byte_end,
                ),
                usage.id.as_str(),
            )
            .is_some()
        {
            return Err(Error::InvalidInput(
                "historical usage annotations duplicate the same occurrence".into(),
            ));
        }
    }
    let mut candidates = Vec::new();
    let mut total_tokens = 0;
    for scene in &manuscript.scenes {
        let text = &scene.text;
        let mut active: Vec<(usize, usize, u32)> = Vec::new();
        let mut previous_end = 0;
        let mut previous_utf16 = 0u32;
        for (token_start, token) in historical_tokens(text) {
            total_tokens += 1;
            if total_tokens > MAX_HISTORICAL_TOKENS {
                return Err(Error::InvalidInput(
                    "historical source token budget exceeded".into(),
                ));
            }
            let utf16_start =
                previous_utf16 + text[previous_end..token_start].encode_utf16().count() as u32;
            let utf16_end = utf16_start + token.encode_utf16().count() as u32;
            let spaced = !text[previous_end..token_start].is_empty();
            let token_end = token_start + token.len();
            previous_end = token_end;
            previous_utf16 = utf16_end;
            if token.len() > MAX_HISTORICAL_TERM_BYTES {
                active.clear();
                continue;
            }
            let word = token.to_lowercase();
            let mut next: Vec<_> = active
                .iter()
                .filter_map(|(node, start, utf16)| {
                    nodes[*node]
                        .children
                        .get(&(spaced, word.clone()))
                        .map(|child| (*child, *start, *utf16))
                })
                .collect();
            if let Some(node) = nodes[0].children.get(&(false, word)) {
                next.push((*node, token_start, utf16_start));
            }
            for (node, start, utf16) in &next {
                for &index in &nodes[*node].entries {
                    if candidates.len() >= MAX_HISTORICAL_CANDIDATES
                        || token_end - start > MAX_HISTORICAL_MATCH_BYTES
                    {
                        return Err(Error::InvalidInput(
                            "historical candidate count or span exceeds limit".into(),
                        ));
                    }
                    let entry = &catalogue[index];
                    let author_usage_id = confirmed
                        .remove(&(entry.id.as_str(), scene.id.clone(), *start, token_end))
                        .map(str::to_string);
                    let date_conflict =
                        (!entry.sense_ambiguous || author_usage_id.is_some()).then(|| {
                            i64::from(entry.earliest_year)
                                > i64::from(target_year) + i64::from(tolerance_years)
                        });
                    candidates.push(HistoricalCandidate {
                        entry_id: entry.id.clone(),
                        scene_id: scene.id.clone(),
                        range: TextRange {
                            byte_start: *start,
                            byte_end: token_end,
                            utf16_start: *utf16,
                            utf16_end,
                        },
                        quote: text[*start..token_end].into(),
                        date_conflict,
                        author_usage_id,
                    });
                }
            }
            active = next;
        }
    }
    if !confirmed.is_empty() {
        return Err(Error::InvalidInput(
            "historical usage does not align with a whole located term".into(),
        ));
    }
    candidates.sort_unstable_by(|a, b| {
        a.scene_id
            .cmp(&b.scene_id)
            .then_with(|| a.range.byte_start.cmp(&b.range.byte_start))
            .then_with(|| a.range.byte_end.cmp(&b.range.byte_end))
            .then_with(|| a.entry_id.cmp(&b.entry_id))
    });
    let mut catalogue = catalogue.to_vec();
    catalogue.sort_unstable_by(|a, b| a.id.cmp(&b.id));
    Ok(HistoricalAnalysisResult { document_id: manuscript.document_id.clone(), target_year, tolerance_years,
        catalogue, candidates, availability: AnalysisAvailability::Partial,
        diagnostics: vec!["caller_catalogue_dates_and_context_not_independently_verified; matches_are_review_candidates".into(),
            "first_attestation_is_not_first_use; source_annotations_confirm_location_and_declared_sense_only".into(),
            "integer_year_constraints_not_calendar_conversion_or_inferred_scene_chronology".into()] })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modern_word_in_medieval() {
        let tokens: Vec<String> = vec!["The", "knight", "checked", "his", "smartphone"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1200, 0);
        assert!(results.iter().any(|a| a.word == "smartphone"));
        assert_eq!(
            results
                .iter()
                .find(|a| a.word == "smartphone")
                .unwrap()
                .introduced_year,
            2007
        );
    }

    #[test]
    fn test_word_within_grace_period() {
        // "telephone" introduced 1876; target 1860 + 20 grace = 1880, so not anachronistic
        let tokens: Vec<String> = vec!["telephone"].into_iter().map(String::from).collect();
        let results = analyze_scene(&tokens, 1860, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_word_before_grace_period() {
        // "telephone" introduced 1876; target 1800 + 20 grace = 1820, so anachronistic
        let tokens: Vec<String> = vec!["telephone"].into_iter().map(String::from).collect();
        let results = analyze_scene(&tokens, 1800, 0);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_multi_word_term() {
        let tokens: Vec<String> = vec!["He", "used", "social", "media", "daily"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1900, 0);
        assert!(results.iter().any(|a| a.word == "social media"));
    }

    #[test]
    fn test_no_target_year() {
        let tokens: Vec<String> = vec!["smartphone"].into_iter().map(String::from).collect();
        let results = analyze_scene(&tokens, 0, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_case_insensitive() {
        let tokens: Vec<String> = vec!["INTERNET"].into_iter().map(String::from).collect();
        let results = analyze_scene(&tokens, 1500, 0);
        assert!(results.iter().any(|a| a.word == "INTERNET"));
    }

    #[test]
    fn test_no_false_positives_for_era() {
        // "atom" introduced 1803; target 1900 should be fine
        let tokens: Vec<String> = vec!["atom", "electron", "vaccine"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1900, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_medical_instruments() {
        let tokens: Vec<String> = vec!["stethoscope", "defibrillator"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1700, 0);
        assert!(results.iter().any(|a| a.word == "stethoscope"));
        assert!(results.iter().any(|a| a.word == "defibrillator"));
    }

    #[test]
    fn test_weapons() {
        let tokens: Vec<String> = vec!["gatling", "bayonet", "napalm"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1600, 0);
        assert!(results.iter().any(|a| a.word == "gatling"));
        assert!(results.iter().any(|a| a.word == "napalm"));
        // bayonet introduced 1670, target 1600 + 20 grace = 1620, so anachronistic
        assert!(results.iter().any(|a| a.word == "bayonet"));
    }

    #[test]
    fn test_clothing() {
        let tokens: Vec<String> = vec!["jeans", "bikini"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1800, 0);
        assert!(results.iter().any(|a| a.word == "jeans"));
        assert!(results.iter().any(|a| a.word == "bikini"));
    }

    #[test]
    fn test_music_terms() {
        let tokens: Vec<String> = vec!["phonograph", "synthesizer"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1800, 0);
        assert!(results.iter().any(|a| a.word == "phonograph"));
        assert!(results.iter().any(|a| a.word == "synthesizer"));
    }

    #[test]
    fn test_sports() {
        let tokens: Vec<String> = vec!["basketball", "volleyball"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1800, 0);
        assert!(results.iter().any(|a| a.word == "basketball"));
        assert!(results.iter().any(|a| a.word == "volleyball"));
    }

    #[test]
    fn test_food() {
        let tokens: Vec<String> = vec!["hamburger", "espresso"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1800, 0);
        assert!(results.iter().any(|a| a.word == "hamburger"));
        assert!(results.iter().any(|a| a.word == "espresso"));
    }

    #[test]
    fn test_slang_cool_in_medieval() {
        let tokens: Vec<String> = vec!["That", "is", "cool"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene_checked(&tokens, 1200, 0).unwrap();
        assert!(results.items.is_empty());
        assert!(results.abstained_terms.iter().any(|term| term == "cool"));
    }

    #[test]
    fn test_slang_okay_before_1839() {
        let tokens: Vec<String> = vec!["okay"].into_iter().map(String::from).collect();
        let results = analyze_scene_checked(&tokens, 1700, 0).unwrap();
        assert!(results.items.is_empty());
        assert!(results.abstained_terms.iter().any(|term| term == "okay"));
    }

    #[test]
    fn test_slang_awesome_modern() {
        let tokens: Vec<String> = vec!["awesome"].into_iter().map(String::from).collect();
        let results = analyze_scene_checked(&tokens, 1900, 0).unwrap();
        assert!(results.items.is_empty());
        assert!(results.abstained_terms.iter().any(|term| term == "awesome"));
    }

    #[test]
    fn test_medieval_concept_warnings() {
        let tokens: Vec<String> = vec!["The", "kingdom", "had", "a", "democracy"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene_checked(&tokens, 1200, 0).unwrap();
        assert!(results.items.is_empty());
        assert_eq!(results.abstained_terms, ["democracy"]);
    }

    #[test]
    fn test_medieval_psychology_warning() {
        let tokens: Vec<String> = vec!["psychology"].into_iter().map(String::from).collect();
        let results = analyze_scene_checked(&tokens, 1100, 0).unwrap();
        assert!(results.items.is_empty());
        assert_eq!(results.abstained_terms, ["psychology"]);
    }

    #[test]
    fn test_medieval_concepts_not_flagged_in_modern() {
        // Medieval-only concepts should not be flagged for modern settings
        let tokens: Vec<String> = vec!["democracy", "evolution", "psychology"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1900, 0);
        // These words are not in the regular milestones, so should not appear
        assert!(!results.iter().any(|a| a.word.contains("democracy")));
    }

    #[test]
    fn test_new_multi_word_terms() {
        let tokens: Vec<String> = vec!["She", "used", "a", "sewing", "machine"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1700, 0);
        assert!(results.iter().any(|a| a.word == "sewing machine"));
    }

    #[test]
    fn test_electronics_terms() {
        let tokens: Vec<String> = vec!["semiconductor", "microprocessor"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1900, 0);
        assert!(results.iter().any(|a| a.word == "semiconductor"));
        assert!(results.iter().any(|a| a.word == "microprocessor"));
    }

    #[test]
    fn test_total_milestone_count() {
        let single = milestones().len();
        let multi = multi_word_milestones().len();
        let slang = slang_milestones().len();
        let medieval = medieval_anachronisms().len();
        // Verify we have substantially more entries than before
        assert!(
            single >= 180,
            "Expected 180+ single-word entries, got {}",
            single
        );
        assert!(
            multi >= 25,
            "Expected 25+ multi-word entries, got {}",
            multi
        );
        assert!(slang >= 10, "Expected 10+ slang entries, got {}", slang);
        assert!(
            medieval >= 20,
            "Expected 20+ medieval entries, got {}",
            medieval
        );
    }
}
