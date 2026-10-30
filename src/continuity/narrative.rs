//! Person mentions, coreference clusters, and quotation speakers from the
//! narrative multi-task model (`ModelKind::NarrativeMultitask`).
//!
//! Two encoder passes per text: the first collects proper and nominal person
//! mentions from the token heads, the second scores antecedents over those
//! mentions plus the crate's pronoun inventory and scores a speaker for every
//! quotation. Antecedent links are decoded greedily inside each 400-token
//! window, then clusters that share an identical proper-name surface are merged
//! across windows, and a final pass runs the antecedent head over the list of
//! distinct proper surfaces so aliases of one character ("Elizabeth", "Miss
//! Bennet") merge too. The heuristic resolvers in [`super::coreference`] and
//! [`crate::craft::dialogue`] call [`analyze`] first and keep their own logic as
//! the fallback when no model is configured.

use std::collections::HashMap;

use crate::onnx::{
    self, InferenceBackend, NerSpans,
    checked::{self, ModelFailure, ModelKind, NarrativeRequest},
    model::ModelIdentity,
};

/// Probability the alias pass needs before two proper surfaces are merged into one
/// character; below it the surfaces stay separate clusters.
pub const ALIAS_MIN_CONFIDENCE: f64 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MentionKind {
    Proper,
    Nominal,
    Pronoun,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NarrativeMention {
    pub start: usize,
    pub end: usize,
    pub kind: MentionKind,
    /// Index into [`NarrativeAnalysis::cluster_names`].
    pub cluster: usize,
    /// The mention the model linked this one to, if any.
    pub antecedent: Option<usize>,
    /// Softmax probability of the chosen antecedent (or of "no antecedent").
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NarrativeQuote {
    pub start: usize,
    pub end: usize,
    /// Index into [`NarrativeAnalysis::mentions`] of the chosen speaker mention.
    pub speaker: Option<usize>,
    /// Softmax probability of the chosen speaker (or of "speaker outside the window").
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct NarrativeAnalysis {
    pub identity: ModelIdentity,
    /// Every entity span the entity head produced (all labels), in text order.
    pub entities: NerSpans,
    /// Mention candidates in `(start, end)` order.
    pub mentions: Vec<NarrativeMention>,
    /// One entry per requested quotation, in the caller's order.
    pub quotes: Vec<NarrativeQuote>,
    /// The most frequent proper-name surface of each cluster, else its most
    /// frequent nominal surface; `None` for a cluster of pronouns only.
    pub cluster_names: Vec<Option<String>>,
    pub windows: usize,
}

impl NarrativeAnalysis {
    pub fn mention_name(&self, mention: usize) -> Option<&str> {
        self.cluster_names[self.mentions[mention].cluster].as_deref()
    }

    pub fn speaker_name(&self, quote: usize) -> Option<&str> {
        self.quotes[quote]
            .speaker
            .and_then(|mention| self.mention_name(mention))
    }
}

/// Whether the process-wide backend has a narrative model ready.
pub fn available() -> bool {
    onnx::backend()
        .check_model(ModelKind::NarrativeMultitask)
        .is_ok()
}

pub fn analyze(text: &str, quotes: &[(usize, usize)]) -> Result<NarrativeAnalysis, ModelFailure> {
    analyze_with_backend(onnx::backend(), text, quotes)
}

pub fn analyze_with_backend(
    backend: &dyn InferenceBackend,
    text: &str,
    quotes: &[(usize, usize)],
) -> Result<NarrativeAnalysis, ModelFailure> {
    let generation = onnx::model_generation();
    let identity = backend.model_identity(ModelKind::NarrativeMultitask)?;
    identity.validate(ModelKind::NarrativeMultitask)?;
    let mut order: Vec<usize> = (0..quotes.len()).collect();
    order.sort_by_key(|&index| quotes[index]);
    let sorted_quotes: Vec<_> = order.iter().map(|&index| quotes[index]).collect();

    let first = checked::narrative(
        backend,
        &NarrativeRequest {
            text,
            mentions: &[],
            quotes: &[],
        },
    )?;
    let mut entities = Vec::new();
    let mut candidates: Vec<(usize, usize, MentionKind)> = Vec::new();
    for window in &first {
        for entity in &window.entities {
            if entity.1 == "PER" {
                candidates.push((entity.2, entity.3, MentionKind::Proper));
            }
            entities.push(entity.clone());
        }
        candidates.extend(
            window
                .nominals
                .iter()
                .map(|&(start, end)| (start, end, MentionKind::Nominal)),
        );
    }
    candidates.extend(
        super::coreference::pronoun_spans(text)
            .map(|(start, end)| (start, end, MentionKind::Pronoun)),
    );
    candidates.sort_unstable();
    candidates.dedup_by_key(|candidate| (candidate.0, candidate.1));
    if candidates.len() > checked::NARRATIVE_SPAN_LIMIT {
        return Err(ModelFailure::execution("narrative_span_limit"));
    }
    let spans: Vec<(usize, usize)> = candidates
        .iter()
        .map(|candidate| (candidate.0, candidate.1))
        .collect();

    let second = checked::narrative(
        backend,
        &NarrativeRequest {
            text,
            mentions: &spans,
            quotes: &sorted_quotes,
        },
    )?;
    if onnx::model_generation() != generation {
        return Err(ModelFailure::execution(
            "model_replaced_during_narrative_analysis",
        ));
    }

    let mut mentions: Vec<NarrativeMention> = candidates
        .iter()
        .map(|&(start, end, kind)| NarrativeMention {
            start,
            end,
            kind,
            cluster: 0,
            antecedent: None,
            confidence: 1.0,
        })
        .collect();
    let mut quote_results: Vec<NarrativeQuote> = sorted_quotes
        .iter()
        .map(|&(start, end)| NarrativeQuote {
            start,
            end,
            speaker: None,
            confidence: 1.0,
        })
        .collect();
    // Overlapping windows score a mention twice; the decision from the window
    // with more left context wins, and a quote keeps the window it sits most
    // centrally in.
    let mut mention_choice: Vec<Option<(usize, Option<usize>, f64)>> = vec![None; mentions.len()];
    let mut quote_choice: Vec<Option<(usize, Option<usize>, f64)>> =
        vec![None; sorted_quotes.len()];
    for window in &second {
        for (local, &global) in window.mentions.iter().enumerate() {
            let row = &window.antecedents[local];
            let (column, probability) = argmax_probability(&row[..=local], |_| true);
            let context = spans[global].0.saturating_sub(window.start);
            if mention_choice[global].is_none_or(|(best, _, _)| context >= best) {
                let antecedent = (column > 0).then(|| window.mentions[column - 1]);
                mention_choice[global] = Some((context, antecedent, probability));
            }
        }
        for (local, &global) in window.quotes.iter().enumerate() {
            let (quote_start, quote_end) = sorted_quotes[global];
            let outside = |column: usize| {
                column == 0 || {
                    let (start, end) = spans[window.mentions[column - 1]];
                    !(start >= quote_start && end <= quote_end)
                }
            };
            let (column, probability) = argmax_probability(&window.speakers[local], outside);
            let centrality = quote_start
                .saturating_sub(window.start)
                .min(window.end.saturating_sub(quote_end));
            if quote_choice[global].is_none_or(|(best, _, _)| centrality >= best) {
                let speaker = (column > 0).then(|| window.mentions[column - 1]);
                quote_choice[global] = Some((centrality, speaker, probability));
            }
        }
    }
    let mut parent: Vec<usize> = (0..mentions.len()).collect();
    for (global, choice) in mention_choice.into_iter().enumerate() {
        if let Some((_, antecedent, probability)) = choice {
            mentions[global].antecedent = antecedent;
            mentions[global].confidence = probability;
            if let Some(antecedent) = antecedent {
                union(&mut parent, global, antecedent);
            }
        }
    }
    for (global, choice) in quote_choice.into_iter().enumerate() {
        if let Some((_, speaker, probability)) = choice {
            quote_results[global].speaker = speaker;
            quote_results[global].confidence = probability;
        }
    }
    // Identical proper-name surfaces name one character across windows.
    let mut first_by_surface: HashMap<&str, usize> = HashMap::new();
    for (index, mention) in mentions.iter().enumerate() {
        if mention.kind == MentionKind::Proper {
            let surface = text[mention.start..mention.end].trim();
            match first_by_surface.get(surface) {
                Some(&first) => union(&mut parent, index, first),
                None => {
                    first_by_surface.insert(surface, index);
                }
            }
        }
    }
    // Alias pass: the antecedent head over the distinct proper surfaces listed in
    // first-appearance order ("Elizabeth . Miss Bennet . Mr. Darcy"); a confident
    // link merges the two surfaces' clusters.
    let mut distinct: Vec<&str> = first_by_surface.keys().copied().collect();
    distinct.sort_by_key(|surface| first_by_surface[surface]);
    if distinct.len() >= 2 {
        let mut alias_text = String::new();
        let mut alias_spans = Vec::with_capacity(distinct.len());
        for (index, surface) in distinct.iter().enumerate() {
            if index > 0 {
                alias_text.push_str(" . ");
            }
            let start = alias_text.len();
            alias_text.push_str(surface);
            alias_spans.push((start, alias_text.len()));
        }
        let alias_windows = checked::narrative(
            backend,
            &NarrativeRequest {
                text: &alias_text,
                mentions: &alias_spans,
                quotes: &[],
            },
        )?;
        for window in &alias_windows {
            for (local, &global) in window.mentions.iter().enumerate() {
                let row = &window.antecedents[local];
                let (column, probability) = argmax_probability(&row[..=local], |_| true);
                if column > 0 && probability >= ALIAS_MIN_CONFIDENCE {
                    let other = window.mentions[column - 1];
                    union(
                        &mut parent,
                        first_by_surface[distinct[global]],
                        first_by_surface[distinct[other]],
                    );
                }
            }
        }
    }
    // A cluster is named by its most frequent proper surface, or, for a character
    // the text never names ("the old man"), by its most frequent nominal surface.
    let mut cluster_of_root: HashMap<usize, usize> = HashMap::new();
    let mut surfaces: Vec<HashMap<&str, (bool, usize, usize)>> = Vec::new();
    for index in 0..mentions.len() {
        let root = find(&mut parent, index);
        let next = cluster_of_root.len();
        let cluster = *cluster_of_root.entry(root).or_insert(next);
        if cluster == surfaces.len() {
            surfaces.push(HashMap::new());
        }
        mentions[index].cluster = cluster;
        if mentions[index].kind != MentionKind::Pronoun {
            let proper = mentions[index].kind == MentionKind::Proper;
            let surface = text[mentions[index].start..mentions[index].end].trim();
            let entry = surfaces[cluster]
                .entry(surface)
                .or_insert((proper, 0, index));
            entry.0 |= proper;
            entry.1 += 1;
        }
    }
    let cluster_names = surfaces
        .iter()
        .map(|counts| {
            counts
                .iter()
                .max_by(|a, b| {
                    (a.1.0, a.1.1, a.0.len(), std::cmp::Reverse(a.1.2)).cmp(&(
                        b.1.0,
                        b.1.1,
                        b.0.len(),
                        std::cmp::Reverse(b.1.2),
                    ))
                })
                .map(|(surface, _)| (*surface).to_string())
        })
        .collect();
    let mut quotes_out = vec![None; quotes.len()];
    for (sorted_index, &original) in order.iter().enumerate() {
        quotes_out[original] = Some(quote_results[sorted_index].clone());
    }
    Ok(NarrativeAnalysis {
        identity,
        entities,
        mentions,
        quotes: quotes_out
            .into_iter()
            .map(|quote| quote.expect("every quote scored"))
            .collect(),
        cluster_names,
        windows: second.len(),
    })
}

/// The best valid column and its softmax probability among the valid columns.
/// Column 0 is always valid, so the result is never empty.
fn argmax_probability(scores: &[f32], valid: impl Fn(usize) -> bool) -> (usize, f64) {
    let mut best = (0usize, f64::NEG_INFINITY);
    let mut valid_scores = Vec::with_capacity(scores.len());
    for (column, &score) in scores.iter().enumerate() {
        if column == 0 || valid(column) {
            let score = f64::from(score);
            valid_scores.push(score);
            if score > best.1 {
                best = (column, score);
            }
        }
    }
    let total: f64 = valid_scores
        .iter()
        .map(|score| (score - best.1).exp())
        .sum();
    (best.0, if total > 0.0 { 1.0 / total } else { 0.0 })
}

fn find(parent: &mut [usize], mut index: usize) -> usize {
    while parent[index] != index {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    index
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    parent[ra.max(rb)] = ra.min(rb);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::checked::NarrativeWindow;

    /// Scripted model: pass one (no mentions) reports the given entity and
    /// nominal spans; pass two links every non-proper mention to the latest
    /// proper mention before it in the same window and names the latest proper
    /// mention before each quote as its speaker.
    struct Scripted {
        entities: Vec<(String, String, usize, usize)>,
        nominals: Vec<(usize, usize)>,
        /// Byte boundaries that split the text into windows.
        splits: Vec<usize>,
    }
    impl InferenceBackend for Scripted {
        fn check_model(&self, _: ModelKind) -> Result<(), ModelFailure> {
            Ok(())
        }
        fn embed(&self, _: &[&str]) -> Vec<Vec<f64>> {
            Vec::new()
        }
        fn tag(&self, _: &str) -> NerSpans {
            Vec::new()
        }
        fn narrative_checked(
            &self,
            request: &NarrativeRequest<'_>,
        ) -> Result<Vec<NarrativeWindow>, ModelFailure> {
            let mut bounds = vec![0];
            bounds.extend(
                self.splits
                    .iter()
                    .map(|split| (*split).min(request.text.len())),
            );
            bounds.push(request.text.len());
            Ok(bounds
                .windows(2)
                .map(|pair| {
                    let (start, end) = (pair[0], pair[1]);
                    let inside = |spans: &[(usize, usize)]| -> Vec<usize> {
                        spans
                            .iter()
                            .enumerate()
                            .filter(|&(_, &(s, e))| s >= start && e <= end)
                            .map(|(index, _)| index)
                            .collect()
                    };
                    let mentions = inside(request.mentions);
                    let quotes = inside(request.quotes);
                    let proper = |index: usize| {
                        let (start, end) = request.mentions[index];
                        self.entities
                            .iter()
                            .any(|entity| entity.0 == request.text[start..end])
                    };
                    let antecedents = (0..mentions.len())
                        .map(|i| {
                            let mut row = vec![0.0; mentions.len() + 1];
                            if !proper(mentions[i])
                                && let Some(j) = (0..i).rev().find(|&j| proper(mentions[j]))
                            {
                                row[j + 1] = 4.0;
                            }
                            row
                        })
                        .collect();
                    let speakers = quotes
                        .iter()
                        .map(|&q| {
                            let mut row = vec![0.0; mentions.len() + 1];
                            let (qs, _) = request.quotes[q];
                            if let Some(j) = (0..mentions.len()).rev().find(|&j| {
                                request.mentions[mentions[j]].1 <= qs && proper(mentions[j])
                            }) {
                                row[j + 1] = 3.0;
                            }
                            row
                        })
                        .collect();
                    NarrativeWindow {
                        start,
                        end,
                        entities: self
                            .entities
                            .iter()
                            .filter(|entity| entity.2 >= start && entity.3 <= end)
                            .cloned()
                            .collect(),
                        nominals: self
                            .nominals
                            .iter()
                            .copied()
                            .filter(|&(s, e)| s >= start && e <= end)
                            .collect(),
                        mentions,
                        quotes,
                        antecedents,
                        speakers,
                    }
                })
                .collect())
        }
    }

    fn at(text: &str, needle: &str, occurrence: usize) -> (usize, usize) {
        let start = text
            .match_indices(needle)
            .nth(occurrence)
            .expect("needle present")
            .0;
        (start, start + needle.len())
    }

    #[test]
    fn mentions_cluster_through_antecedents_and_quotes_name_their_speaker() {
        let text = "Alice smiled at her mother. \"Come in,\" she said.";
        let alice = at(text, "Alice", 0);
        let mother = at(text, "her mother", 0);
        let quote = at(text, "\"Come in,\"", 0);
        let backend = Scripted {
            entities: vec![("Alice".into(), "PER".into(), alice.0, alice.1)],
            nominals: vec![mother],
            splits: vec![],
        };
        let analysis = analyze_with_backend(&backend, text, &[quote]).unwrap();
        assert_eq!(analysis.windows, 1);
        let kinds: Vec<_> = analysis
            .mentions
            .iter()
            .map(|mention| (&text[mention.start..mention.end], mention.kind))
            .collect();
        assert_eq!(
            kinds,
            [
                ("Alice", MentionKind::Proper),
                ("her", MentionKind::Pronoun),
                ("her mother", MentionKind::Nominal),
                ("she", MentionKind::Pronoun),
            ]
        );
        assert!(analysis.mentions.iter().all(|mention| mention.cluster == 0));
        assert_eq!(analysis.mentions[3].antecedent, Some(0));
        assert!(analysis.mentions[3].confidence > 0.9);
        assert_eq!(analysis.cluster_names, [Some("Alice".to_string())]);
        assert_eq!(analysis.speaker_name(0), Some("Alice"));
        assert_eq!(analysis.quotes[0].speaker, Some(0));
    }

    #[test]
    fn identical_proper_surfaces_merge_clusters_across_windows_and_quote_order_is_kept() {
        let text =
            "Alice waved. Bob nodded. | Alice laughed and she left. \"Go,\" Bob said. \"Stay,\"";
        let split = text.find('|').unwrap();
        let alice1 = at(text, "Alice", 0);
        let bob1 = at(text, "Bob", 0);
        let alice2 = at(text, "Alice", 1);
        let bob2 = at(text, "Bob", 1);
        let go = at(text, "\"Go,\"", 0);
        let stay = at(text, "\"Stay,\"", 0);
        let per = |(start, end): (usize, usize)| {
            (text[start..end].to_string(), "PER".to_string(), start, end)
        };
        let backend = Scripted {
            entities: vec![per(alice1), per(bob1), per(alice2), per(bob2)],
            nominals: vec![],
            splits: vec![split],
        };
        let analysis = analyze_with_backend(&backend, text, &[stay, go]).unwrap();
        assert_eq!(analysis.windows, 2);
        let cluster = |span: (usize, usize)| {
            analysis
                .mentions
                .iter()
                .find(|mention| (mention.start, mention.end) == span)
                .unwrap()
                .cluster
        };
        assert_eq!(cluster(alice1), cluster(alice2));
        assert_eq!(cluster(bob1), cluster(bob2));
        assert_ne!(cluster(alice1), cluster(bob1));
        let she = at(text, "she", 0);
        assert_eq!(cluster(she), cluster(alice2));
        assert_eq!(analysis.cluster_names.len(), 2);
        // The scripted speaker is the last proper mention before the quote.
        assert_eq!(analysis.speaker_name(1), Some("Alice"));
        assert_eq!(analysis.speaker_name(0), Some("Bob"));
        assert_eq!((analysis.quotes[0].start, analysis.quotes[0].end), stay);
    }

    #[test]
    fn argmax_probability_masks_columns_and_normalizes_over_valid_ones() {
        assert_eq!(argmax_probability(&[0.0], |_| true), (0, 1.0));
        let (column, probability) = argmax_probability(&[0.0, 5.0, 1.0], |column| column != 1);
        assert_eq!(column, 2);
        assert!((probability - 1.0 / (1.0 + (-1.0f64).exp())).abs() < 1e-12);
    }
}
