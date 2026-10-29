//! Entity resolution: cluster name mentions into canonical entities using
//! layered, local, deterministic evidence — no per-operation LLM.
//!
//! The robustness key is that surface evidence is the *gate*, not a free merge.
//! Surfaces are compatible only when their normalized token sets are equal or one
//! is a subset of the other (after honorific/article stripping and nickname
//! normalization of the given name) — so a bare "David" / "Mr. Condrey" folds
//! into "David Condrey", while "David Condrey" and "Bob Condrey" (different given
//! names sharing only a surname) stay separate on surface alone, even with no
//! context. **Context-embedding similarity** from the self-rolled corpus
//! embedding then confirms genuinely ambiguous attachments (a lone "David" that
//! could join more than one "David X"). When no context vector is available it
//! degrades to the surface gate.
//!
//! Mentions carry confidence rather than being dropped, and every entity carries
//! [`Provenance`]; user decisions ([`Provenance::UserEdited`]) are authoritative
//! and survive re-resolution.

use serde::{Deserialize, Serialize};

use crate::substrate::common_types::Provenance;

/// What kind of thing an entity is. Characters, settings, and symbols share one
/// resolution mechanism.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    Character,
    Setting,
    Symbol,
    Other,
}

/// One occurrence of a name in the text, with the embedding of its surrounding
/// context (empty if unavailable).
#[derive(Debug, Clone)]
pub struct Mention {
    pub surface: String,
    pub context: Vec<f64>,
}

impl Mention {
    pub fn new(surface: impl Into<String>, context: Vec<f64>) -> Self {
        Mention {
            surface: surface.into(),
            context,
        }
    }
}

/// A resolved entity: its canonical form, observed surface forms, and evidence.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Entity {
    pub id: String,
    pub canonical: String,
    pub kind: EntityType,
    pub aliases: Vec<String>,
    pub mention_count: usize,
    /// Heuristic support tier, not a calibrated semantic-accuracy probability.
    pub confidence: f64,
    pub provenance: Provenance,
}

/// Minimum context cosine for two surface-compatible mentions to be the same
/// entity. Below it, ambiguous surnames resolve to separate entities.
const DEFAULT_CONTEXT_FLOOR: f64 = 0.45;
pub const MAX_ENTITY_MENTIONS: usize = 10_000;
pub const MAX_ENTITY_CONTEXT_VALUES: usize = 1_048_576;

fn validate_surface(surface: &str) -> crate::Result<()> {
    crate::manuscript::config::validate_id(surface)
}
fn validated_mentions(mentions: &[Mention]) -> crate::Result<Vec<Mention>> {
    if mentions.len() > MAX_ENTITY_MENTIONS {
        return Err(crate::Error::InvalidInput(
            "entity mentions exceed10000".into(),
        ));
    }
    let mut values = 0usize;
    let mut dimensions = None;
    let mut normalized = Vec::with_capacity(mentions.len());
    for mention in mentions {
        validate_surface(&mention.surface)?;
        values = values
            .checked_add(mention.context.len())
            .filter(|n| *n <= MAX_ENTITY_CONTEXT_VALUES)
            .ok_or_else(|| crate::Error::InvalidInput("entity context budget exceeded".into()))?;
        if mention.context.len() > 4096 || mention.context.iter().any(|value| !value.is_finite()) {
            return Err(crate::Error::InvalidInput(
                "entity contexts require at most4096 finite values".into(),
            ));
        }
        let scale = mention
            .context
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        let context = if scale == 0.0 {
            Vec::new()
        } else {
            if dimensions.is_some_and(|d| d != mention.context.len()) {
                return Err(crate::Error::InvalidInput(
                    "entity context dimensions disagree".into(),
                ));
            }
            dimensions = Some(mention.context.len());
            let mut context: Vec<_> = mention.context.iter().map(|value| value / scale).collect();
            let norm = context
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            for value in &mut context {
                *value /= norm;
            }
            context
        };
        normalized.push(Mention::new(mention.surface.clone(), context));
    }
    Ok(normalized)
}

pub fn resolve_checked(mentions: &[Mention], kind: EntityType) -> crate::Result<Vec<Entity>> {
    Ok(resolve_with_floor(
        &validated_mentions(mentions)?,
        kind,
        DEFAULT_CONTEXT_FLOOR,
    ))
}

pub fn resolve_with_corrections_checked(
    mentions: &[Mention],
    kind: EntityType,
    corrections: &[EntityCorrection],
) -> crate::Result<Vec<Entity>> {
    let normalized = validated_mentions(mentions)?;
    if corrections.len() > MAX_ENTITY_MENTIONS {
        return Err(crate::Error::InvalidInput(
            "entity corrections exceed10000".into(),
        ));
    }
    let mut bytes = mentions
        .iter()
        .map(|mention| mention.surface.len())
        .sum::<usize>();
    for correction in corrections {
        validate_surface(&correction.surface)?;
        let canonical = if let CorrectionAction::Assign { canonical } = &correction.action {
            validate_surface(canonical)?;
            canonical.len()
        } else {
            0
        };
        bytes = bytes
            .checked_add(correction.surface.len() + canonical)
            .filter(|n| *n <= 4 * 1024 * 1024)
            .ok_or_else(|| {
                crate::Error::InvalidInput("entity correction metadata exceeds4MiB".into())
            })?;
    }
    if corrections.is_empty() {
        return Ok(resolve_with_floor(&normalized, kind, DEFAULT_CONTEXT_FLOOR));
    }
    Ok(apply_corrections(
        resolve_clusters(&normalized, DEFAULT_CONTEXT_FLOOR, kind),
        kind,
        corrections,
    ))
}

fn entity_id<'a>(kind: EntityType, surfaces: impl Iterator<Item = &'a str>) -> String {
    let key = surfaces
        .map(|surface| {
            let mut tokens = normalized_tokens(kind, surface);
            if kind != EntityType::Character {
                tokens.sort();
            }
            tokens
        })
        .max_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)))
        .unwrap_or_default()
        .join("\0");
    let mut hash = 0xcbf29ce484222325u64;
    for byte in format!("entity-v2:{kind:?}\0{key}").bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let slug = crate::substrate::utils::slug(&key.replace('\0', " "))
        .chars()
        .take(64)
        .collect::<String>();
    format!("ent-{slug}-{hash:016x}")
}
fn finalize_ids(entities: &mut [Entity]) {
    entities.sort_by(|a, b| {
        a.canonical
            .cmp(&b.canonical)
            .then_with(|| a.aliases.cmp(&b.aliases))
            .then_with(|| a.mention_count.cmp(&b.mention_count))
            .then_with(|| a.id.cmp(&b.id))
    });
    let mut groups = std::collections::BTreeMap::<String, Vec<usize>>::new();
    for (index, entity) in entities.iter().enumerate() {
        groups.entry(entity.id.clone()).or_default().push(index);
    }
    for (id, indices) in groups {
        if indices.len() > 1 {
            for (suffix, index) in indices.into_iter().enumerate() {
                entities[index].id = format!("{id}-{}", suffix + 1);
            }
        }
    }
}

struct Cluster {
    surfaces: Vec<(String, usize)>, // surface form -> count
    forms: Vec<Vec<String>>,
    centroid: Vec<f64>,
    context_n: usize,
}

/// Resolve mentions into entities of the given `kind`.
pub fn resolve(mentions: &[Mention], kind: EntityType) -> Vec<Entity> {
    resolve_checked(mentions, kind).unwrap_or_default()
}

fn resolve_with_floor(mentions: &[Mention], kind: EntityType, floor: f64) -> Vec<Entity> {
    let mut entities: Vec<_> = resolve_clusters(mentions, floor, kind)
        .into_iter()
        .map(|c| finish_cluster(c, kind))
        .collect();
    finalize_ids(&mut entities);
    entities
}

pub(crate) fn resolve_counted(
    surfaces: &[(String, usize)],
    kind: EntityType,
) -> crate::Result<Vec<Entity>> {
    if surfaces.len() > MAX_ENTITY_MENTIONS {
        return Err(crate::Error::InvalidInput(
            "entity surfaces exceed10000".into(),
        ));
    }
    let mut weights = std::collections::BTreeMap::<String, usize>::new();
    let mut total = 0usize;
    for (surface, count) in surfaces {
        validate_surface(surface)?;
        total = total
            .checked_add(*count)
            .ok_or_else(|| crate::Error::InvalidInput("entity mention counts overflow".into()))?;
        if *count > 0 {
            *weights.entry(surface.clone()).or_default() += count;
        }
    }
    let mentions: Vec<_> = weights
        .keys()
        .map(|surface| Mention::new(surface, vec![]))
        .collect();
    let mut clusters = resolve_clusters(&mentions, DEFAULT_CONTEXT_FLOOR, kind);
    for cluster in &mut clusters {
        for (surface, count) in &mut cluster.surfaces {
            *count = weights[surface];
        }
    }
    let mut entities: Vec<_> = clusters
        .into_iter()
        .map(|cluster| finish_cluster(cluster, kind))
        .collect();
    finalize_ids(&mut entities);
    Ok(entities)
}

fn surface_key(kind: EntityType, surface: &str) -> Vec<String> {
    let mut tokens = normalized_tokens(kind, surface);
    if kind != EntityType::Character {
        tokens.sort();
        tokens.dedup();
    }
    tokens
}
fn compatible_tokens(kind: EntityType, a: &[String], b: &[String]) -> bool {
    if kind == EntityType::Character {
        if a.len() == 1 {
            return a[0] == b[0] || Some(&a[0]) == b.last();
        }
        if b.len() == 1 {
            return b[0] == a[0] || Some(&b[0]) == a.last();
        }
        if a.first() != b.first() || a.last() != b.last() {
            return false;
        }
        let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
        let mut matched = 0;
        for token in long {
            if short.get(matched) == Some(token) {
                matched += 1;
            }
        }
        return matched == short.len();
    }
    a.iter().all(|token| b.binary_search(token).is_ok())
        || b.iter().all(|token| a.binary_search(token).is_ok())
}
fn resolve_clusters(mentions: &[Mention], floor: f64, kind: EntityType) -> Vec<Cluster> {
    let mut ordered: Vec<_> = mentions
        .iter()
        .map(|mention| (mention, surface_key(kind, &mention.surface)))
        .collect();
    ordered.sort_by(|(a, ak), (b, bk)| {
        bk.len()
            .cmp(&ak.len())
            .then_with(|| a.surface.cmp(&b.surface))
            .then_with(|| {
                a.context
                    .iter()
                    .zip(&b.context)
                    .map(|(a, b)| a.total_cmp(b))
                    .find(|order| !order.is_eq())
                    .unwrap_or_else(|| a.context.len().cmp(&b.context.len()))
            })
    });
    let mut clusters: Vec<Cluster> = Vec::new();
    let mut index = std::collections::BTreeMap::<String, std::collections::BTreeSet<usize>>::new();
    for (mention, key) in ordered {
        let postings: Vec<_> = key.iter().filter_map(|token| index.get(token)).collect();
        let possible = if postings.len() == key.len() {
            let mut possible = postings
                .iter()
                .min_by_key(|set| set.len())
                .map(|set| (*set).clone())
                .unwrap_or_default();
            possible.retain(|candidate| postings.iter().all(|set| set.contains(candidate)));
            possible
        } else {
            std::collections::BTreeSet::new()
        };
        let candidates: Vec<_> = possible
            .into_iter()
            .filter(|&i| cluster_accepts(&clusters[i], mention, floor, &key, kind))
            .collect();
        let exact: Vec<_> = candidates
            .iter()
            .copied()
            .filter(|&i| clusters[i].forms.contains(&key))
            .collect();
        let matched = if exact.len() == 1 {
            Some(exact[0])
        } else if candidates.len() == 1 {
            Some(candidates[0])
        } else {
            None
        };
        let id = if let Some(id) = matched {
            add_to_cluster(&mut clusters[id], mention, &key);
            id
        } else {
            clusters.push(new_cluster(mention, &key));
            clusters.len() - 1
        };
        for token in key {
            index.entry(token).or_default().insert(id);
        }
    }
    clusters
}

/// A writer's authoritative decision about one mention surface, persisted and
/// reapplied over the automatic resolution. A correction is keyed by the
/// `surface` form it targets; the caller looks corrections up per document.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntityCorrection {
    /// The mention surface this decision applies to (e.g. "Mr. Condrey").
    pub surface: String,
    /// What the writer decided for that surface.
    pub action: CorrectionAction,
}

/// The three writer mutations the entity surface supports. `Tag` and `Reassign`
/// both bind a surface to a named canonical entity (tagging a previously
/// unrecognized mention vs. moving a mis-grouped one); they carry the same data
/// and are applied identically, so they share [`CorrectionAction::Assign`].
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CorrectionAction {
    /// Bind the surface to the entity with this canonical name (tag or reassign).
    Assign { canonical: String },
    /// Suppress the surface: it is not a real entity mention.
    Reject,
}

impl<'de> Deserialize<'de> for CorrectionAction {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
        enum Wire {
            Assign { canonical: String },
            Reject {},
        }
        Ok(match Wire::deserialize(deserializer)? {
            Wire::Assign { canonical } => Self::Assign { canonical },
            Wire::Reject {} => Self::Reject,
        })
    }
}

/// Resolve mentions, then apply the writer's persisted `corrections` over the
/// automatic result. Corrections are authoritative: a rejected surface is
/// dropped, an assigned surface is moved to (or used to create) the named
/// canonical entity. Reuses [`resolve`] verbatim — corrections are a
/// deterministic post-pass, never a re-run of the resolution algorithm.
pub fn resolve_with_corrections(
    mentions: &[Mention],
    kind: EntityType,
    corrections: &[EntityCorrection],
) -> Vec<Entity> {
    resolve_with_corrections_checked(mentions, kind, corrections).unwrap_or_default()
}

/// Apply corrections to an already-resolved entity set. Split out so a caller
/// can also surface a correction's effect deterministically.
fn apply_corrections(
    clusters: Vec<Cluster>,
    kind: EntityType,
    corrections: &[EntityCorrection],
) -> Vec<Entity> {
    // Index corrections by surface; a later correction for the same surface wins
    // (the writer's most recent decision is authoritative).
    let mut by_surface: std::collections::BTreeMap<&str, &CorrectionAction> =
        std::collections::BTreeMap::new();
    for c in corrections {
        by_surface.insert(c.surface.as_str(), &c.action);
    }

    // Re-bin every auto surface per its correction. An assigned surface joins a
    // bin keyed by the writer's chosen canonical name (which is *forced* as that
    // bin's canonical, never re-derived from surface frequency); an unassigned
    // surface stays in its original auto entity; a rejected surface is dropped.
    // An entity left with no surfaces disappears.
    let mut bins: Vec<Bin> = Vec::new();
    let mut bin_index: std::collections::HashMap<BinKey, usize> = std::collections::HashMap::new();

    let mut clusters = clusters;
    for cluster in &mut clusters {
        sort_surfaces(&mut cluster.surfaces, kind);
    }
    let mut counts = std::collections::BTreeMap::new();
    for cluster in &clusters {
        *counts
            .entry(cluster.surfaces[0].0.clone())
            .or_insert(0usize) += 1;
    }
    for (cluster_id, cluster) in clusters.into_iter().enumerate() {
        let surfaces = cluster.surfaces;
        let auto_canonical = surfaces[0].0.clone();
        for (surface, count) in surfaces {
            match by_surface.get(surface.as_str()) {
                Some(CorrectionAction::Reject) => {}
                Some(CorrectionAction::Assign { canonical }) => {
                    place_assigned(&mut bins, &mut bin_index, kind, canonical, surface, count);
                }
                None => place_auto(
                    &mut bins,
                    &mut bin_index,
                    kind,
                    if counts[&auto_canonical] == 1 {
                        BinKey::Canonical(auto_canonical.clone())
                    } else {
                        BinKey::Auto(cluster_id)
                    },
                    surface,
                    count,
                ),
            }
        }
    }

    // A tag may name a surface the resolver never produced (a missed mention).
    // Materialize an assigned surface that landed in no bin.
    let mut placed: std::collections::BTreeSet<String> = bins
        .iter()
        .flat_map(|bin| bin.surfaces.iter().map(|(surface, _)| surface.clone()))
        .collect();
    for (surface, action) in by_surface {
        if let CorrectionAction::Assign { canonical } = action
            && !placed.contains(surface)
        {
            place_assigned(
                &mut bins,
                &mut bin_index,
                kind,
                canonical,
                surface.to_string(),
                1,
            );
            placed.insert(surface.into());
        }
    }

    let mut entities: Vec<_> = bins.into_iter().map(finish_bin).collect();
    finalize_ids(&mut entities);
    entities
}

/// One regrouped entity-in-progress. `forced_canonical` is set when a writer
/// assigned the surface to a named entity, pinning that name as canonical.
#[derive(Hash, PartialEq, Eq)]
enum BinKey {
    Canonical(String),
    Auto(usize),
}
struct Bin {
    kind: EntityType,
    forced_canonical: Option<String>,
    surfaces: Vec<(String, usize)>,
    user_edited: bool,
}

fn bin_for<'a>(
    bins: &'a mut Vec<Bin>,
    bin_index: &mut std::collections::HashMap<BinKey, usize>,
    key: BinKey,
    kind: EntityType,
) -> &'a mut Bin {
    let i = *bin_index.entry(key).or_insert_with(|| {
        bins.push(Bin {
            kind,
            forced_canonical: None,
            surfaces: Vec::new(),
            user_edited: false,
        });
        bins.len() - 1
    });
    &mut bins[i]
}

fn place_auto(
    bins: &mut Vec<Bin>,
    bin_index: &mut std::collections::HashMap<BinKey, usize>,
    kind: EntityType,
    key: BinKey,
    surface: String,
    count: usize,
) {
    let bin = bin_for(bins, bin_index, key, kind);
    bin.surfaces.push((surface, count));
}

fn place_assigned(
    bins: &mut Vec<Bin>,
    bin_index: &mut std::collections::HashMap<BinKey, usize>,
    kind: EntityType,
    canonical: &str,
    surface: String,
    count: usize,
) {
    // Key by the canonical name itself — the same keyspace `place_auto` uses —
    // so assigning a surface to an entity that auto-resolution already produced
    // (canonical names equal) merges into that one entity rather than emitting a
    // second entity with the same canonical and the same id.
    let bin = bin_for(
        bins,
        bin_index,
        BinKey::Canonical(canonical.to_string()),
        kind,
    );
    bin.forced_canonical = Some(canonical.to_string());
    bin.user_edited = true;
    bin.surfaces.push((surface, count));
}

/// Finish a regrouped bin into an [`Entity`]. A forced canonical (from an assign)
/// is used verbatim; otherwise the canonical is derived from surfaces exactly as
/// after auto-resolution.
fn finish_bin(bin: Bin) -> Entity {
    // A bin can collect the same surface from more than one source entity (e.g.
    // two auto entities sharing a canonical, or a canonical surface plus an
    // assigned one of the same form); merge them so a surface is counted once and
    // never appears as a duplicate alias.
    let surfaces = merge_surfaces(bin.surfaces);
    match bin.forced_canonical {
        Some(canonical) => finish_forced(canonical, surfaces, bin.kind),
        None => {
            let mut ent = finish_cluster(
                Cluster {
                    surfaces,
                    forms: Vec::new(),
                    centroid: Vec::new(),
                    context_n: 0,
                },
                bin.kind,
            );
            if bin.user_edited {
                ent.provenance = Provenance::UserEdited;
            }
            ent
        }
    }
}

/// Sum counts for repeated surface strings, preserving first-seen order.
fn merge_surfaces(surfaces: Vec<(String, usize)>) -> Vec<(String, usize)> {
    let mut order: Vec<String> = Vec::new();
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (s, n) in surfaces {
        if !counts.contains_key(&s) {
            order.push(s.clone());
        }
        *counts.entry(s).or_insert(0) += n;
    }
    order
        .into_iter()
        .map(|s| {
            let n = counts[&s];
            (s, n)
        })
        .collect()
}

/// Build an entity whose canonical is fixed by a writer assignment. The chosen
/// name is canonical even if it never appeared as an auto mention; every other
/// distinct surface becomes an alias. Counts are the real observed counts (no
/// synthetic inflation), so `mention_count` reflects actual mentions.
fn finish_forced(canonical: String, surfaces: Vec<(String, usize)>, kind: EntityType) -> Entity {
    let mention_count: usize = surfaces.iter().map(|(_, n)| n).sum::<usize>().max(1);
    let aliases: Vec<String> = surfaces
        .into_iter()
        .map(|(s, _)| s)
        .filter(|s| s != &canonical)
        .collect();
    // Distinct surface forms = the canonical plus each alias.
    let forms = aliases.len() + 1;
    Entity {
        id: entity_id(kind, std::iter::once(canonical.as_str())),
        canonical: canonical.clone(),
        kind,
        aliases,
        mention_count,
        confidence: confidence_of(mention_count, forms),
        provenance: Provenance::UserEdited,
    }
}

/// A mention joins a cluster when it is surface-compatible with any of the
/// cluster's known forms AND (when both have context) context-similar.
fn cluster_accepts(
    cluster: &Cluster,
    mention: &Mention,
    floor: f64,
    key: &[String],
    kind: EntityType,
) -> bool {
    if !cluster
        .forms
        .iter()
        .any(|form| compatible_tokens(kind, form, key))
        || key.len() > 1
            && cluster
                .forms
                .iter()
                .any(|form| form.len() > 1 && !compatible_tokens(kind, form, key))
    {
        return false;
    }
    cluster.context_n == 0
        || mention.context.is_empty()
        || crate::substrate::utils::cosine_similarity(&cluster.centroid, &mention.context) >= floor
}

fn new_cluster(m: &Mention, key: &[String]) -> Cluster {
    let (centroid, context_n) = if m.context.is_empty() {
        (Vec::new(), 0)
    } else {
        (m.context.clone(), 1)
    };
    Cluster {
        surfaces: vec![(m.surface.clone(), 1)],
        forms: vec![key.to_vec()],
        centroid,
        context_n,
    }
}

fn add_to_cluster(cluster: &mut Cluster, m: &Mention, key: &[String]) {
    if !cluster.forms.iter().any(|form| form == key) {
        cluster.forms.push(key.to_vec());
    }
    match cluster.surfaces.iter_mut().find(|(s, _)| s == &m.surface) {
        Some((_, n)) => *n += 1,
        None => cluster.surfaces.push((m.surface.clone(), 1)),
    }
    if !m.context.is_empty() {
        if cluster.centroid.is_empty() {
            cluster.centroid = m.context.clone();
        } else if cluster.centroid.len() == m.context.len() {
            // Running mean of context vectors.
            let n = cluster.context_n as f64;
            for (c, x) in cluster.centroid.iter_mut().zip(&m.context) {
                *c = (*c * n + x) / (n + 1.0);
            }
        }
        cluster.context_n += 1;
    }
}

/// Canonical first: the most frequent surface, except that a character's
/// fuller name with at least half that support outranks a bare given name
/// ("Hester Prynne" over "Hester", which every full mention also counts as).
fn sort_surfaces(surfaces: &mut [(String, usize)], kind: EntityType) {
    let top = surfaces.iter().map(|(_, count)| *count).max().unwrap_or(0);
    let rank = |surface: &str, count: usize| {
        let full = kind == EntityType::Character && count.saturating_mul(2) >= top;
        (full.then(|| surface.split_whitespace().count()), count)
    };
    surfaces.sort_by(|a, b| {
        rank(&b.0, b.1)
            .cmp(&rank(&a.0, a.1))
            .then_with(|| b.0.len().cmp(&a.0.len()))
            .then_with(|| a.0.cmp(&b.0))
    });
}

fn finish_cluster(cluster: Cluster, kind: EntityType) -> Entity {
    let mut surfaces = cluster.surfaces;
    sort_surfaces(&mut surfaces, kind);
    let mention_count: usize = surfaces.iter().map(|(_, n)| n).sum();
    let canonical = surfaces[0].0.clone();
    let id = entity_id(kind, surfaces.iter().map(|(surface, _)| surface.as_str()));
    let aliases: Vec<String> = surfaces.iter().skip(1).map(|(s, _)| s.clone()).collect();
    let confidence = confidence_of(mention_count, surfaces.len());
    Entity {
        id,
        canonical,
        kind,
        aliases,
        mention_count,
        confidence,
        provenance: Provenance::AiDerived,
    }
}

/// Confidence rises with mentions and corroborating surface forms; even a single
/// mention keeps a small positive confidence (surfaced, not discarded).
fn confidence_of(mentions: usize, forms: usize) -> f64 {
    let base = (mentions as f64).min(5.0) / 5.0; // saturates at 5 mentions
    let corroboration = ((forms.saturating_sub(1)) as f64 * 0.1).min(0.3);
    crate::substrate::utils::round4(((base * 0.7) + corroboration).min(1.0))
}

/// Honorifics stripped before surface comparison so "Mr. Condrey" matches "David
/// Condrey" on the shared "Condrey".
const HONORIFICS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sir", "lady", "lord", "king", "queen", "captain", "colonel",
    "sergeant", "aunt", "uncle", "father", "mother",
];

/// Kind-aware comparison tokens preserve surnames, setting names and symbol
/// punctuation; only leading character titles and given-name nicknames fold.
fn normalized_tokens(kind: EntityType, surface: &str) -> Vec<String> {
    if kind == EntityType::Symbol {
        let mut words: Vec<_> = surface.split_whitespace().map(str::to_lowercase).collect();
        if words.len() > 1 && matches!(words[0].as_str(), "the" | "a" | "an") {
            words.remove(0);
        }
        return vec![words.join(" ")];
    }
    let mut tokens: Vec<_> = surface
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|part| !part.is_empty())
        .map(str::to_lowercase)
        .collect();
    if tokens.len() > 1
        && (tokens[0] == "the"
            || kind != EntityType::Character && matches!(tokens[0].as_str(), "a" | "an"))
    {
        tokens.remove(0);
    }
    if kind == EntityType::Character {
        while tokens.len() > 1 && tokens[1] != "s" && HONORIFICS.contains(&tokens[0].as_str()) {
            tokens.remove(0);
        }
        if let Some(first) = tokens.first_mut() {
            *first = nickname_key(first);
        }
    }
    if tokens.is_empty() {
        tokens.push(format!("@raw:{}", surface.trim().to_lowercase()));
    }
    tokens
}

/// Canonical key for a given/nickname form (Bob and Robert both map to robert).
/// A built-in seed; this is exactly the kind of data that should move to an
/// external lexicon file (see ARCHITECTURE notes on externalizing data).
fn nickname_key(name: &str) -> String {
    let n = name.to_lowercase();
    let canon = match n.as_str() {
        "bob" | "rob" | "robbie" | "bobby" => "robert",
        "bill" | "billy" | "will" | "willie" => "william",
        "liz" | "lizzy" | "beth" | "betty" | "eliza" => "elizabeth",
        "dick" | "rick" | "ricky" | "richie" => "richard",
        "jim" | "jimmy" | "jamie" => "james",
        "tom" | "tommy" => "thomas",
        "kate" | "katie" | "kathy" | "katherine" => "catherine",
        "peggy" | "meg" | "maggie" => "margaret",
        "tony" => "anthony",
        "ed" | "eddie" | "ned" => "edward",
        other => other,
    };
    canon.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(surface: &str, ctx: &[f64]) -> Mention {
        Mention::new(surface, ctx.to_vec())
    }

    #[test]
    fn merges_aliases_of_one_character() {
        let ctx = [1.0, 0.0];
        let ents = resolve(
            &[
                m("David Condrey", &ctx),
                m("David", &[0.95, 0.05]),
                m("Mr. Condrey", &[0.9, 0.1]),
            ],
            EntityType::Character,
        );
        assert_eq!(ents.len(), 1, "{ents:?}");
        assert_eq!(ents[0].canonical, "David Condrey");
        assert!(ents[0].aliases.iter().any(|a| a == "David"));
        assert!(ents[0].aliases.iter().any(|a| a == "Mr. Condrey"));
        assert_eq!(ents[0].mention_count, 3);
    }

    #[test]
    fn same_surname_different_context_stays_separate() {
        // Both share "Condrey" but live in different contexts → two entities.
        let ents = resolve(
            &[
                m("David Condrey", &[1.0, 0.0]),
                m("Bob Condrey", &[0.0, 1.0]),
            ],
            EntityType::Character,
        );
        assert_eq!(
            ents.len(),
            2,
            "should not merge distinct Condreys: {ents:?}"
        );
    }

    #[test]
    fn nickname_and_full_name_merge_in_same_context() {
        let ents = resolve(
            &[m("Robert", &[1.0, 0.0]), m("Bob", &[0.97, 0.03])],
            EntityType::Character,
        );
        assert_eq!(ents.len(), 1, "Bob/Robert should merge: {ents:?}");
    }

    #[test]
    fn single_mention_is_surfaced_with_low_confidence() {
        let ents = resolve(&[m("Eleanor", &[1.0, 0.0])], EntityType::Character);
        assert_eq!(ents.len(), 1);
        assert!(
            ents[0].confidence > 0.0 && ents[0].confidence < 0.5,
            "{:?}",
            ents[0]
        );
    }

    #[test]
    fn resolves_settings_the_same_way_as_characters() {
        // Same mechanism, EntityType::Setting: aliases of one place merge;
        // a different place stays separate (proving it's not name-only).
        let ents = resolve(
            &[
                m("the Castle", &[1.0, 0.0]),
                m("Castle", &[0.96, 0.04]),
                m("the Harbor", &[0.0, 1.0]),
            ],
            EntityType::Setting,
        );
        assert_eq!(ents.len(), 2, "{ents:?}");
        assert!(ents.iter().all(|e| e.kind == EntityType::Setting));
        let castle = ents
            .iter()
            .find(|e| e.canonical.to_lowercase().contains("castle"))
            .unwrap();
        assert_eq!(castle.mention_count, 2);
    }

    #[test]
    fn string_only_fallback_without_context() {
        let ents = resolve(
            &[m("David Condrey", &[]), m("David", &[])],
            EntityType::Character,
        );
        assert_eq!(ents.len(), 1, "no context → string-only merge: {ents:?}");
    }

    #[test]
    fn reject_correction_drops_a_resolved_entity() {
        let mentions = [m("Eleanor", &[1.0, 0.0]), m("Eleanor", &[0.99, 0.01])];
        let corrections = [EntityCorrection {
            surface: "Eleanor".into(),
            action: CorrectionAction::Reject,
        }];
        let ents = resolve_with_corrections(&mentions, EntityType::Character, &corrections);
        assert!(
            ents.is_empty(),
            "rejected surface should not resolve: {ents:?}"
        );
    }

    #[test]
    fn reassign_correction_splits_a_mismerged_entity() {
        // Auto-resolution folds "Castle" into "the Castle"; a reassign moves it to
        // a separate canonical the writer names.
        let mentions = [m("the Castle", &[1.0, 0.0]), m("Castle", &[0.99, 0.01])];
        let auto = resolve(&mentions, EntityType::Setting);
        assert_eq!(auto.len(), 1, "auto should merge: {auto:?}");
        let corrections = [EntityCorrection {
            surface: "Castle".into(),
            action: CorrectionAction::Assign {
                canonical: "Castle Ruins".into(),
            },
        }];
        let ents = resolve_with_corrections(&mentions, EntityType::Setting, &corrections);
        assert_eq!(ents.len(), 2, "reassign should split: {ents:?}");
        let moved = ents.iter().find(|e| e.canonical == "Castle Ruins").unwrap();
        assert_eq!(moved.provenance, Provenance::UserEdited);
    }

    #[test]
    fn tag_correction_materializes_an_unseen_surface() {
        // No mentions resolve to "Narrator"; tagging it creates the entity.
        let mentions = [m("David", &[1.0, 0.0])];
        let corrections = [EntityCorrection {
            surface: "Narrator".into(),
            action: CorrectionAction::Assign {
                canonical: "The Narrator".into(),
            },
        }];
        let ents = resolve_with_corrections(&mentions, EntityType::Character, &corrections);
        assert!(
            ents.iter().any(|e| e.canonical == "The Narrator"),
            "tagged surface should appear: {ents:?}"
        );
    }

    #[test]
    fn different_given_names_sharing_surname_stay_separate_string_only() {
        // The production path supplies no context, so this must hold on surface
        // evidence alone: two people who share only a surname are not one entity.
        let ents = resolve(
            &[m("John Smith", &[]), m("Mary Smith", &[])],
            EntityType::Character,
        );
        assert_eq!(
            ents.len(),
            2,
            "siblings sharing a surname must not merge: {ents:?}"
        );
    }

    #[test]
    fn shared_given_name_different_surname_stays_separate_string_only() {
        let ents = resolve(
            &[m("David Brown", &[]), m("David Green", &[])],
            EntityType::Character,
        );
        assert_eq!(ents.len(), 2, "{ents:?}");
    }

    #[test]
    fn substring_names_do_not_merge() {
        // "Ann" is a substring of "Anne" but a different name; token-wise they
        // differ, so they must not collapse.
        let ents = resolve(&[m("Ann", &[]), m("Anne", &[])], EntityType::Character);
        assert_eq!(ents.len(), 2, "{ents:?}");
    }

    #[test]
    fn nickname_with_shared_surname_merges_string_only() {
        // Same surname + nickname-equivalent given name = one person, even without
        // context.
        let ents = resolve(
            &[m("Bob Condrey", &[]), m("Robert Condrey", &[])],
            EntityType::Character,
        );
        assert_eq!(ents.len(), 1, "Bob/Robert Condrey are one person: {ents:?}");
    }

    #[test]
    fn reassign_to_existing_canonical_merges_into_it() {
        // "David" and "Dave" auto-resolve separately (no shared token, no
        // context). Assigning "Dave" to the existing canonical "David" must fold
        // it into that one entity — not emit a second entity also named "David".
        let mentions = [m("David", &[]), m("Dave", &[])];
        let auto = resolve(&mentions, EntityType::Character);
        assert_eq!(
            auto.len(),
            2,
            "precondition: should auto-resolve apart: {auto:?}"
        );
        let corrections = [EntityCorrection {
            surface: "Dave".into(),
            action: CorrectionAction::Assign {
                canonical: "David".into(),
            },
        }];
        let ents = resolve_with_corrections(&mentions, EntityType::Character, &corrections);
        assert_eq!(
            ents.len(),
            1,
            "assign to existing canonical must not duplicate: {ents:?}"
        );
        assert_eq!(ents[0].canonical, "David");
        assert!(ents[0].aliases.iter().any(|a| a == "Dave"), "{ents:?}");
        assert_eq!(
            ents[0].mention_count, 2,
            "both mentions counted once: {ents:?}"
        );
        assert_eq!(ents[0].provenance, Provenance::UserEdited);
    }

    #[test]
    fn assigned_entity_reports_honest_mention_count() {
        // Tagging one unseen surface yields exactly one mention — no synthetic
        // inflation from the forced-canonical injection.
        let mentions = [m("David", &[])];
        let corrections = [EntityCorrection {
            surface: "Narrator".into(),
            action: CorrectionAction::Assign {
                canonical: "The Narrator".into(),
            },
        }];
        let ents = resolve_with_corrections(&mentions, EntityType::Character, &corrections);
        let narrator = ents
            .iter()
            .find(|e| e.canonical == "The Narrator")
            .unwrap_or_else(|| panic!("tagged entity missing: {ents:?}"));
        assert_eq!(
            narrator.mention_count, 1,
            "one mention, not inflated: {narrator:?}"
        );
        assert_eq!(narrator.aliases, vec!["Narrator".to_string()]);
    }

    #[test]
    fn counted_resolution_checks_allocation_independent_count_bounds() {
        assert!(
            resolve_counted(
                &[("Alice".into(), usize::MAX), ("Bob".into(), 1)],
                EntityType::Character
            )
            .is_err()
        );
        for count in [usize::MAX - 1, usize::MAX] {
            let entities =
                resolve_counted(&[("Alice".into(), count)], EntityType::Character).unwrap();
            assert_eq!(entities[0].mention_count, count);
        }
        let entities = resolve_counted(
            &[("John Smith".into(), 3), ("John".into(), 7)],
            EntityType::Character,
        )
        .unwrap();
        assert_eq!(entities[0].canonical, "John");
        assert_eq!(entities[0].mention_count, 10);
    }

    #[test]
    fn correction_roundtrips_through_json() {
        let c = EntityCorrection {
            surface: "Mr. Condrey".into(),
            action: CorrectionAction::Assign {
                canonical: "David Condrey".into(),
            },
        };
        let bytes = serde_json::to_vec(&c).unwrap();
        assert_eq!(
            serde_json::from_slice::<EntityCorrection>(&bytes).unwrap(),
            c
        );
    }
}
