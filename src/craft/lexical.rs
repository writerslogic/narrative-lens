//! Sparse lexical retrieval (Okapi BM25) and rank fusion.
//!
//! Dense embeddings miss exact and rare terms: a name occurring once is dropped
//! from the embedding vocabulary (`min_count`), so a query for it has no dense
//! signal. BM25 scores exact term overlap and covers that blind spot. The two
//! rankings are combined with reciprocal rank fusion, which is scale-free (it
//! uses ranks, not the incomparable cosine/BM25 score magnitudes) and so needs
//! no per-corpus threshold tuning.

use std::collections::{BTreeSet, HashMap};

/// BM25 term-frequency saturation.
const K1: f64 = 1.5;
/// BM25 length-normalization strength.
const B: f64 = 0.75;
/// Reciprocal-rank-fusion damping; 60 is the value from the original RRF paper.
pub const RRF_K: f64 = 60.0;

/// An Okapi BM25 index over a set of `(id, text)` units (documents or scenes).
#[derive(Debug, Clone, Default)]
pub struct Bm25Index {
    ids: Vec<String>,
    /// term → postings of `(unit position, term frequency)`.
    postings: HashMap<String, Vec<(usize, usize)>>,
    doc_len: Vec<usize>,
    avgdl: f64,
}

impl Bm25Index {
    /// Build over `(id, text)` units, tokenizing exactly as the embedding does
    /// (content tokens, stopwords removed) so dense and lexical see the same
    /// terms.
    pub fn build(units: &[(String, String)]) -> Bm25Index {
        let stop = crate::substrate::text::stopwords();
        let mut ids = Vec::with_capacity(units.len());
        let mut doc_len = Vec::with_capacity(units.len());
        let mut postings: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
        for (pos, (id, text)) in units.iter().enumerate() {
            ids.push(id.clone());
            let mut tf: HashMap<String, usize> = HashMap::new();
            let mut len = 0usize;
            for tok in crate::substrate::text::content_tokens(text, stop) {
                *tf.entry(tok).or_insert(0) += 1;
                len += 1;
            }
            doc_len.push(len);
            for (term, count) in tf {
                postings.entry(term).or_default().push((pos, count));
            }
        }
        let total: u64 = doc_len.iter().map(|&l| l as u64).sum();
        let avgdl = if doc_len.is_empty() {
            0.0
        } else {
            total as f64 / doc_len.len() as f64
        };
        Bm25Index {
            ids,
            postings,
            doc_len,
            avgdl,
        }
    }

    pub fn update_document(&mut self, id: &str, text: &str) {
        let position = self
            .ids
            .iter()
            .position(|candidate| candidate == id)
            .unwrap_or_else(|| {
                self.ids.push(id.to_owned());
                self.doc_len.push(0);
                self.ids.len() - 1
            });
        let mut frequencies = HashMap::<String, usize>::new();
        let mut length = 0;
        for token in
            crate::substrate::text::content_tokens(text, crate::substrate::text::stopwords())
        {
            *frequencies.entry(token).or_default() += 1;
            length += 1;
        }
        self.postings.retain(|_, postings| {
            postings.retain(|(document, _)| *document != position);
            !postings.is_empty()
        });
        for (term, frequency) in frequencies {
            self.postings
                .entry(term)
                .or_default()
                .push((position, frequency));
        }
        self.doc_len[position] = length;
        self.avgdl =
            self.doc_len.iter().map(|&value| value as f64).sum::<f64>() / self.ids.len() as f64;
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Units ranked by BM25 relevance to `query`, strongest first; only units
    /// with a positive score are returned.
    pub fn search(&self, query: &str, limit: usize) -> Vec<(String, f64)> {
        if self.ids.is_empty() || self.avgdl <= 0.0 {
            return Vec::new();
        }
        let stop = crate::substrate::text::stopwords();
        let n = self.ids.len() as f64;
        let mut scores = vec![0.0f64; self.ids.len()];
        let terms: BTreeSet<String> = crate::substrate::text::content_tokens(query, stop)
            .into_iter()
            .collect();
        for term in terms {
            let Some(postings) = self.postings.get(&term) else {
                continue;
            };
            let df = postings.len() as f64;
            // BM25 idf with the +1 guard, so it never goes negative.
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &(pos, tf) in postings {
                let tf = tf as f64;
                let dl = self.doc_len[pos] as f64;
                let denom = tf + K1 * (1.0 - B + B * dl / self.avgdl);
                scores[pos] += idf * (tf * (K1 + 1.0)) / denom;
            }
        }
        let mut hits: Vec<(String, f64)> = scores
            .iter()
            .enumerate()
            .filter(|(_, s)| **s > 0.0)
            .map(|(i, s)| (self.ids[i].clone(), *s))
            .collect();
        hits.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        hits.truncate(limit);
        hits
    }
}

/// Combine ranked id lists (best first) with reciprocal rank fusion: each list
/// contributes `1 / (k + rank)` to an id's score. Returns the fused ranking,
/// strongest first, truncated to `limit`.
pub fn reciprocal_rank_fusion(
    rankings: &[Vec<String>],
    k: f64,
    limit: usize,
) -> Vec<(String, f64)> {
    reciprocal_rank_fusion_checked(rankings, k, limit).unwrap_or_default()
}

pub fn reciprocal_rank_fusion_checked(
    rankings: &[Vec<String>],
    k: f64,
    limit: usize,
) -> crate::Result<Vec<(String, f64)>> {
    if !k.is_finite() || k < 0.0 || rankings.len() > 10_000 {
        return Err(crate::Error::InvalidInput(
            "RRF requires finite nonnegative k and at most10000 rankings".into(),
        ));
    }
    let mut entries = 0usize;
    for ranking in rankings {
        entries = entries
            .checked_add(ranking.len())
            .filter(|count| *count <= 100_000)
            .ok_or_else(|| crate::Error::InvalidInput("RRF exceeds100000 ranked entries".into()))?;
        for id in ranking {
            crate::manuscript::config::validate_id(id)?;
        }
    }
    let mut fused: HashMap<String, f64> = HashMap::new();
    for ranking in rankings {
        let mut seen = std::collections::HashSet::new();
        for id in ranking {
            if seen.insert(id) {
                *fused.entry(id.clone()).or_insert(0.0) += 1.0 / (k + seen.len() as f64);
            }
        }
    }
    let mut out: Vec<(String, f64)> = fused.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out.truncate(limit);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn units() -> Vec<(String, String)> {
        vec![
            (
                "d1".into(),
                "the cat sat on the warm mat by the fire".into(),
            ),
            (
                "d2".into(),
                "a dog ran across the muddy field at dawn".into(),
            ),
            (
                "d3".into(),
                "the zarathustra manuscript lay hidden in the vault".into(),
            ),
        ]
    }

    #[test]
    fn bm25_ranks_exact_term_match_first() {
        let index = Bm25Index::build(&units());
        let hits = index.search("zarathustra", 3);
        assert_eq!(hits[0].0, "d3", "hits: {hits:?}");
    }

    #[test]
    fn bm25_empty_query_returns_nothing() {
        let index = Bm25Index::build(&units());
        assert!(index.search("", 3).is_empty());
    }

    #[test]
    fn rrf_rewards_agreement_across_rankings() {
        // "b" is ranked highly by both lists; "a" tops one but is absent from the
        // other. Fusion should put the agreed-upon "b" first.
        let dense = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let lexical = vec!["b".to_string(), "d".to_string()];
        let fused = reciprocal_rank_fusion(&[dense, lexical], RRF_K, 4);
        assert_eq!(fused[0].0, "b", "fused: {fused:?}");
    }
}
