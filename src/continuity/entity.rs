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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub canonical: String,
    pub kind: EntityType,
    pub aliases: Vec<String>,
    pub mention_count: usize,
    /// 0..1; low for thinly-evidenced entities (surfaced, not dropped).
    pub confidence: f64,
    pub provenance: Provenance,
}

/// Minimum context cosine for two surface-compatible mentions to be the same
/// entity. Below it, ambiguous surnames resolve to separate entities.
const DEFAULT_CONTEXT_FLOOR: f64 = 0.45;

struct Cluster {
    surfaces: Vec<(String, usize)>, // surface form -> count
    centroid: Vec<f64>,
    context_n: usize,
}

/// Resolve mentions into entities of the given `kind`.
pub fn resolve(mentions: &[Mention], kind: EntityType) -> Vec<Entity> {
    resolve_with_floor(mentions, kind, DEFAULT_CONTEXT_FLOOR)
}

fn resolve_with_floor(mentions: &[Mention], kind: EntityType, floor: f64) -> Vec<Entity> {
    let mut clusters: Vec<Cluster> = Vec::new();

    for m in mentions {
        let mut matched = None;
        for (i, c) in clusters.iter().enumerate() {
            if cluster_accepts(c, m, floor) {
                matched = Some(i);
                break;
            }
        }
        match matched {
            Some(i) => add_to_cluster(&mut clusters[i], m),
            None => clusters.push(new_cluster(m)),
        }
    }

    clusters
        .into_iter()
        .map(|c| finish_cluster(c, kind))
        .collect()
}

/// A writer's authoritative decision about one mention surface, persisted and
/// reapplied over the automatic resolution. A correction is keyed by the
/// `surface` form it targets; the caller looks corrections up per document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CorrectionAction {
    /// Bind the surface to the entity with this canonical name (tag or reassign).
    Assign { canonical: String },
    /// Suppress the surface: it is not a real entity mention.
    Reject,
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
    let auto = resolve(mentions, kind);
    apply_corrections(auto, kind, corrections)
}

/// Apply corrections to an already-resolved entity set. Split out so a caller
/// can also surface a correction's effect deterministically.
fn apply_corrections(
    entities: Vec<Entity>,
    kind: EntityType,
    corrections: &[EntityCorrection],
) -> Vec<Entity> {
    if corrections.is_empty() {
        return entities;
    }
    // Index corrections by surface; a later correction for the same surface wins
    // (the writer's most recent decision is authoritative).
    let mut by_surface: std::collections::HashMap<&str, &CorrectionAction> =
        std::collections::HashMap::new();
    for c in corrections {
        by_surface.insert(c.surface.as_str(), &c.action);
    }

    // Re-bin every auto surface per its correction. An assigned surface joins a
    // bin keyed by the writer's chosen canonical name (which is *forced* as that
    // bin's canonical, never re-derived from surface frequency); an unassigned
    // surface stays in its original auto entity; a rejected surface is dropped.
    // An entity left with no surfaces disappears.
    let mut bins: Vec<Bin> = Vec::new();
    let mut bin_index: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    for ent in entities {
        // Reconstruct per-surface counts: aliases carry no count in `Entity`, so
        // give the canonical the bulk of the mentions and each alias one.
        let alias_n = ent.aliases.len();
        let canonical_count = ent.mention_count.saturating_sub(alias_n).max(1);
        let surfaces = std::iter::once((ent.canonical.clone(), canonical_count))
            .chain(ent.aliases.iter().cloned().map(|a| (a, 1usize)));
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
                    &ent.canonical,
                    surface,
                    count,
                ),
            }
        }
    }

    // A tag may name a surface the resolver never produced (a missed mention).
    // Materialize an assigned surface that landed in no bin.
    for c in corrections {
        if let CorrectionAction::Assign { canonical } = &c.action
            && !bins.iter().any(|b| b.surfaces.iter().any(|(s, _)| s == &c.surface))
        {
            place_assigned(&mut bins, &mut bin_index, kind, canonical, c.surface.clone(), 1);
        }
    }

    bins.into_iter().map(finish_bin).collect()
}

/// One regrouped entity-in-progress. `forced_canonical` is set when a writer
/// assigned the surface to a named entity, pinning that name as canonical.
struct Bin {
    kind: EntityType,
    forced_canonical: Option<String>,
    surfaces: Vec<(String, usize)>,
    user_edited: bool,
}

fn bin_for<'a>(
    bins: &'a mut Vec<Bin>,
    bin_index: &mut std::collections::HashMap<String, usize>,
    key: String,
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
    bin_index: &mut std::collections::HashMap<String, usize>,
    kind: EntityType,
    auto_canonical: &str,
    surface: String,
    count: usize,
) {
    let bin = bin_for(bins, bin_index, auto_canonical.to_string(), kind);
    bin.surfaces.push((surface, count));
}

fn place_assigned(
    bins: &mut Vec<Bin>,
    bin_index: &mut std::collections::HashMap<String, usize>,
    kind: EntityType,
    canonical: &str,
    surface: String,
    count: usize,
) {
    // Key by the canonical name itself — the same keyspace `place_auto` uses —
    // so assigning a surface to an entity that auto-resolution already produced
    // (canonical names equal) merges into that one entity rather than emitting a
    // second entity with the same canonical and the same id.
    let bin = bin_for(bins, bin_index, canonical.to_string(), kind);
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
        id: format!("ent-{}", crate::substrate::utils::slug(&canonical)),
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
fn cluster_accepts(cluster: &Cluster, m: &Mention, floor: f64) -> bool {
    let surface_ok = cluster
        .surfaces
        .iter()
        .any(|(s, _)| surfaces_compatible(s, &m.surface));
    if !surface_ok {
        return false;
    }
    if cluster.context_n == 0 || m.context.is_empty() {
        return true; // no context to disambiguate on → string-only
    }
    crate::substrate::utils::cosine_similarity(&cluster.centroid, &m.context) >= floor
}

fn new_cluster(m: &Mention) -> Cluster {
    let (centroid, context_n) = if m.context.is_empty() {
        (Vec::new(), 0)
    } else {
        (m.context.clone(), 1)
    };
    Cluster {
        surfaces: vec![(m.surface.clone(), 1)],
        centroid,
        context_n,
    }
}

fn add_to_cluster(cluster: &mut Cluster, m: &Mention) {
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

fn finish_cluster(cluster: Cluster, kind: EntityType) -> Entity {
    let mut surfaces = cluster.surfaces;
    // Canonical = most frequent surface; longest breaks ties (fuller name).
    surfaces.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.len().cmp(&a.0.len())));
    let mention_count: usize = surfaces.iter().map(|(_, n)| n).sum();
    let canonical = surfaces[0].0.clone();
    let aliases: Vec<String> = surfaces.iter().skip(1).map(|(s, _)| s.clone()).collect();
    let confidence = confidence_of(mention_count, surfaces.len());
    Entity {
        id: format!("ent-{}", crate::substrate::utils::slug(&canonical)),
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

/// Two surface forms are compatible if, after stripping honorifics/articles and
/// normalizing the given name through the nickname lexicon, their token *sets*
/// are equal or one is a subset of the other.
///
/// Subset — not "shares any token" — is the safe rule when no context is
/// available to disambiguate (the production path supplies none): a bare given
/// name or surname ("David", "Condrey", "Mr. Condrey") attaches to the fuller
/// name it is contained in, while two full names that merely share one token
/// ("John Smith" / "Mary Smith", "David Brown" / "David Green") do **not** merge.
/// Comparison is token-wise, so a substring coincidence ("Ann" / "Anne") is not
/// a match. When context vectors are present, [`cluster_accepts`] still confirms
/// an ambiguous subset attachment against the cluster centroid.
fn surfaces_compatible(a: &str, b: &str) -> bool {
    let aw = normalized_tokens(a);
    let bw = normalized_tokens(b);
    if aw.is_empty() || bw.is_empty() {
        return false;
    }
    let aset: std::collections::HashSet<&str> = aw.iter().map(String::as_str).collect();
    let bset: std::collections::HashSet<&str> = bw.iter().map(String::as_str).collect();
    // `is_subset` is true for equal sets too, so this covers exact matches,
    // reorderings, and bare-name attachment in one test.
    aset.is_subset(&bset) || bset.is_subset(&aset)
}

/// Content tokens with the given (first) token mapped through the nickname
/// lexicon, so "Bob Condrey" and "Robert Condrey" share a token set. Only the
/// leading token is normalized — surnames are left verbatim.
fn normalized_tokens(s: &str) -> Vec<String> {
    let mut tokens = content_tokens(s);
    if let Some(first) = tokens.first_mut() {
        *first = nickname_key(first);
    }
    tokens
}

fn content_tokens(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .filter(|t| !HONORIFICS.contains(&t.as_str()) && !matches!(t.as_str(), "the" | "a" | "an"))
        .collect()
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
        assert!(ents.is_empty(), "rejected surface should not resolve: {ents:?}");
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
        assert_eq!(ents.len(), 2, "siblings sharing a surname must not merge: {ents:?}");
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
        assert_eq!(auto.len(), 2, "precondition: should auto-resolve apart: {auto:?}");
        let corrections = [EntityCorrection {
            surface: "Dave".into(),
            action: CorrectionAction::Assign {
                canonical: "David".into(),
            },
        }];
        let ents = resolve_with_corrections(&mentions, EntityType::Character, &corrections);
        assert_eq!(ents.len(), 1, "assign to existing canonical must not duplicate: {ents:?}");
        assert_eq!(ents[0].canonical, "David");
        assert!(ents[0].aliases.iter().any(|a| a == "Dave"), "{ents:?}");
        assert_eq!(ents[0].mention_count, 2, "both mentions counted once: {ents:?}");
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
        assert_eq!(narrator.mention_count, 1, "one mention, not inflated: {narrator:?}");
        assert_eq!(narrator.aliases, vec!["Narrator".to_string()]);
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
        assert_eq!(serde_json::from_slice::<EntityCorrection>(&bytes).unwrap(), c);
    }
}
