//! Corpus-derived embeddings rolled on the fly from the writer's own text.
//!
//! No pretrained model, no LLM, no network — the semantic space is built
//! entirely from the manuscript's own vocabulary and word co-occurrence, so it
//! captures *this* writer's usage (a thesaurus/semantic index tuned to their
//! book) rather than a generic corpus. Corpus embeddings excel at
//! intra-manuscript similarity and "words like X *in my book*"; a pretrained
//! general-purpose embedding model is better for general semantics.
//!
//! Technique: a **PPMI + truncated SVD** count-based model (latent semantic
//! analysis), which rivals word2vec on intrinsic similarity (Levy & Goldberg,
//! 2014) while staying fully deterministic and model-free. Steps: (1)
//! distance-weighted word co-occurrence over a sliding window; (2) positive
//! pointwise mutual information with context-distribution smoothing (α = 0.75,
//! which downweights frequent words and surfaces informative associations); (3)
//! truncated SVD of the PPMI matrix into dense, denoised word vectors.
//! Deterministic (SVD has no RNG), local, and fast at the capped vocabulary size
//! used here; rebuilt whenever the manuscript changes.

use std::collections::HashMap;

use nalgebra::DMatrix;

/// Tunable build parameters.
#[derive(Debug, Clone)]
pub struct EmbeddingOptions {
    /// Embedding dimensionality.
    pub dim: usize,
    /// Co-occurrence window radius (content tokens on each side).
    pub window: usize,
    /// Keep at most this many vocabulary words (most frequent win).
    pub max_vocab: usize,
    /// Ignore words occurring fewer than this many times.
    pub min_count: usize,
}

impl Default for EmbeddingOptions {
    fn default() -> Self {
        EmbeddingOptions {
            dim: 100,
            window: 5,
            // Randomized truncated SVD keeps the build fast at this size; small
            // vocabularies use an exact SVD (see `truncated_svd`).
            max_vocab: 2500,
            min_count: 2,
        }
    }
}

/// Context-distribution smoothing exponent for PPMI (Levy & Goldberg).
const PPMI_SMOOTHING: f64 = 0.75;

/// Smooth inverse-frequency pooling constant (Arora et al., 2017): a passage
/// vector weights each word by `a / (a + p(word))`, so frequent words contribute
/// less than rare, salient ones instead of every content word counting equally.
const SIF_A: f64 = 1e-3;

/// A semantic space learned from one corpus. Word vectors are L2-normalized.
#[derive(Debug, Clone)]
pub struct CorpusEmbedding {
    dim: usize,
    vectors: HashMap<String, Vec<f64>>,
    /// Smooth inverse-frequency weight per vocabulary word, used for pooling
    /// passages (see [`SIF_A`]).
    weights: HashMap<String, f64>,
}

impl CorpusEmbedding {
    /// Build an embedding from the given texts (e.g. a project's documents).
    pub fn train(texts: &[&str], opts: &EmbeddingOptions) -> CorpusEmbedding {
        // 1. Per-text content-token sequences + global counts.
        let stop = crate::substrate::text::stopwords();
        let sequences: Vec<Vec<String>> = texts
            .iter()
            .map(|t| crate::substrate::text::content_tokens(t, stop))
            .collect();
        let mut counts: HashMap<String, usize> = HashMap::new();
        for seq in &sequences {
            for w in seq {
                *counts.entry(w.clone()).or_insert(0) += 1;
            }
        }

        // 2. Vocabulary: drop rare words, cap to the most frequent, index it.
        let total_tokens: usize = counts.values().sum();
        let mut vocab: Vec<(String, usize)> = counts
            .into_iter()
            .filter(|(_, c)| *c >= opts.min_count)
            .collect();
        vocab.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        vocab.truncate(opts.max_vocab.max(1));
        let words: Vec<String> = vocab.iter().map(|(w, _)| w.clone()).collect();
        let n = words.len();
        if n == 0 {
            return CorpusEmbedding {
                dim: 0,
                vectors: HashMap::new(),
                weights: HashMap::new(),
            };
        }
        // Smooth inverse-frequency pooling weights from corpus frequencies.
        let weights: HashMap<String, f64> = vocab
            .iter()
            .map(|(w, c)| {
                let p = *c as f64 / total_tokens.max(1) as f64;
                (w.clone(), SIF_A / (SIF_A + p))
            })
            .collect();
        let idx: HashMap<&str, usize> = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i))
            .collect();

        // 3. Distance-weighted symmetric co-occurrence matrix.
        let mut cooc = DMatrix::<f64>::zeros(n, n);
        for seq in &sequences {
            for (t, target) in seq.iter().enumerate() {
                let Some(&ti) = idx.get(target.as_str()) else {
                    continue;
                };
                let lo = t.saturating_sub(opts.window);
                let hi = (t + opts.window + 1).min(seq.len());
                for (c, ctx) in seq.iter().enumerate().take(hi).skip(lo) {
                    if c == t {
                        continue;
                    }
                    if let Some(&ci) = idx.get(ctx.as_str()) {
                        cooc[(ti, ci)] += 1.0 / t.abs_diff(c) as f64;
                    }
                }
            }
        }

        // 4. PPMI with context-distribution smoothing, then 5. truncated SVD.
        // The space can hold at most `n` components.
        let dim = opts.dim.max(1).min(n);
        let ppmi = ppmi_matrix(cooc);
        let vectors = svd_vectors(&ppmi, &words, dim);

        CorpusEmbedding {
            dim,
            vectors,
            weights,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn vocab_size(&self) -> usize {
        self.vectors.len()
    }

    /// The learned vector for a word, if it is in the vocabulary.
    pub fn word_vector(&self, word: &str) -> Option<&[f64]> {
        self.vectors.get(&word.to_lowercase()).map(|v| v.as_slice())
    }

    /// Embed an arbitrary passage as the L2-normalized, smooth-inverse-frequency
    /// weighted average of its in-vocabulary content-word vectors: rare, salient
    /// words count more than frequent ones. Returns a zero vector if no word is
    /// known.
    pub fn embed_text(&self, text: &str) -> Vec<f64> {
        let stop = crate::substrate::text::stopwords();
        let mut sum = vec![0.0; self.dim];
        let mut total_weight = 0.0;
        for w in crate::substrate::text::content_tokens(text, stop) {
            if let Some(v) = self.vectors.get(&w) {
                let weight = self.weights.get(&w).copied().unwrap_or(1.0);
                for (s, x) in sum.iter_mut().zip(v) {
                    *s += weight * x;
                }
                total_weight += weight;
            }
        }
        if total_weight > 0.0 {
            for s in sum.iter_mut() {
                *s /= total_weight;
            }
            l2_normalize(&mut sum);
        }
        sum
    }

    /// Cosine similarity between two passages in this manuscript's space.
    pub fn similarity(&self, a: &str, b: &str) -> f64 {
        crate::substrate::utils::cosine_similarity(&self.embed_text(a), &self.embed_text(b))
    }

    /// The `k` vocabulary words most similar to `word` (a manuscript-tuned
    /// thesaurus), strongest first. Empty if the word is unknown.
    pub fn nearest_words(&self, word: &str, k: usize) -> Vec<(String, f64)> {
        let key = word.to_lowercase();
        let Some(target) = self.vectors.get(&key) else {
            return Vec::new();
        };
        let mut scored: Vec<(String, f64)> = self
            .vectors
            .iter()
            .filter(|(w, _)| *w != &key)
            .map(|(w, v)| {
                (
                    w.clone(),
                    crate::substrate::utils::cosine_similarity(target, v),
                )
            })
            .collect();
        crate::substrate::utils::sort_by_score_desc(&mut scored);
        scored.truncate(k);
        scored
    }
}

/// A scene located in its source document, with the UTF-16 code-unit offsets of
/// its content (the unit editors use for edit ranges) so a passage hit can be
/// navigated to directly, plus a short snippet for display.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneRef {
    pub document_id: String,
    pub scene_index: usize,
    pub utf16_start: u32,
    pub utf16_end: u32,
    pub snippet: String,
}

/// Characters kept in a scene's display snippet.
const SNIPPET_CHARS: usize = 120;

fn snippet_of(text: &str) -> String {
    let mut snippet: String = text.chars().take(SNIPPET_CHARS).collect();
    if text.chars().nth(SNIPPET_CHARS).is_some() {
        snippet.push('…');
    }
    snippet
}

/// Every scene of a manuscript embedded in the *one* shared space, so scenes are
/// comparable across documents (passage retrieval, and the motif/redundancy
/// levers built on it). Distinct from per-document scene vectors, which live in
/// incompatible per-document spaces.
#[derive(Debug, Clone, Default)]
pub struct SceneIndex {
    scenes: Vec<SceneRef>,
    vectors: Vec<Vec<f64>>,
}

impl SceneIndex {
    fn build(documents: &[(String, String)], embedding: &CorpusEmbedding) -> SceneIndex {
        let mut index = SceneIndex::default();
        for (doc_id, text) in documents {
            index.insert_document(doc_id, text, embedding);
        }
        index
    }

    fn insert_document(&mut self, doc_id: &str, text: &str, embedding: &CorpusEmbedding) {
        for (i, (scene, start, end)) in crate::substrate::text::paragraph_spans(text)
            .into_iter()
            .enumerate()
        {
            self.vectors.push(embedding.embed_text(&scene));
            self.scenes.push(SceneRef {
                document_id: doc_id.to_string(),
                scene_index: i,
                utf16_start: start,
                utf16_end: end,
                snippet: snippet_of(&scene),
            });
        }
    }

    /// Re-segment and re-embed one document's scenes, leaving every other
    /// document's untouched, so locators stay correct after an edit.
    fn update_document(&mut self, doc_id: &str, text: &str, embedding: &CorpusEmbedding) {
        let mut scenes = Vec::with_capacity(self.scenes.len());
        let mut vectors = Vec::with_capacity(self.vectors.len());
        for (s, v) in std::mem::take(&mut self.scenes)
            .into_iter()
            .zip(std::mem::take(&mut self.vectors))
        {
            if s.document_id != doc_id {
                scenes.push(s);
                vectors.push(v);
            }
        }
        self.scenes = scenes;
        self.vectors = vectors;
        self.insert_document(doc_id, text, embedding);
    }

    fn search(
        &self,
        embedding: &CorpusEmbedding,
        query: &str,
        limit: usize,
    ) -> Vec<(SceneRef, f64)> {
        let q = embedding.embed_text(query);
        let mut hits: Vec<(usize, f64)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, crate::substrate::utils::cosine_similarity(&q, v)))
            .collect();
        crate::substrate::utils::sort_by_score_desc(&mut hits);
        hits.truncate(limit);
        hits.into_iter()
            .map(|(i, s)| (self.scenes[i].clone(), s))
            .collect()
    }

    /// Pairs of *distant* scenes (different documents, or far apart in the same
    /// one — `min_gap` scenes) most similar in the shared space, strongest first.
    /// Surfaces recurring descriptions and repeated beats. Ranked rather than
    /// thresholded, so it needs no embedder-specific similarity floor.
    pub fn redundant_pairs(&self, min_gap: usize, limit: usize) -> Vec<(SceneRef, SceneRef, f64)> {
        let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..self.scenes.len() {
            for j in (i + 1)..self.scenes.len() {
                let distant = self.scenes[i].document_id != self.scenes[j].document_id
                    || self.scenes[j]
                        .scene_index
                        .abs_diff(self.scenes[i].scene_index)
                        >= min_gap;
                if !distant {
                    continue;
                }
                let sim =
                    crate::substrate::utils::cosine_similarity(&self.vectors[i], &self.vectors[j]);
                if sim > 0.0 {
                    pairs.push((i, j, sim));
                }
            }
        }
        pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(limit);
        pairs
            .into_iter()
            .map(|(i, j, sim)| (self.scenes[i].clone(), self.scenes[j].clone(), sim))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.scenes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scenes.is_empty()
    }

    pub fn scenes(&self) -> &[SceneRef] {
        &self.scenes
    }
}

/// A searchable semantic index over a set of documents: the corpus embedding
/// plus each document's cached vector, so repeated queries don't re-embed every
/// document. Built once and reused until the corpus changes. Also holds a
/// [`SceneIndex`] over the same space for passage-level retrieval.
#[derive(Debug, Clone)]
pub struct SemanticIndex {
    embedding: CorpusEmbedding,
    doc_vectors: Vec<(String, Vec<f64>)>,
    scenes: SceneIndex,
    /// Sparse lexical index over documents, fused with the dense ranking so
    /// exact/rare terms are not missed. Reflects the last full build (rebuilt
    /// with the embedding model, not on each incremental edit).
    lexical_docs: crate::craft::lexical::Bm25Index,
}

impl SemanticIndex {
    /// Build from `(document_id, text)` pairs.
    pub fn build(documents: &[(String, String)], opts: &EmbeddingOptions) -> SemanticIndex {
        let texts: Vec<&str> = documents.iter().map(|(_, t)| t.as_str()).collect();
        let embedding = CorpusEmbedding::train(&texts, opts);
        let doc_vectors = documents
            .iter()
            .map(|(id, t)| (id.clone(), embedding.embed_text(t)))
            .collect();
        let scenes = SceneIndex::build(documents, &embedding);
        let lexical_docs = crate::craft::lexical::Bm25Index::build(documents);
        SemanticIndex {
            embedding,
            doc_vectors,
            scenes,
            lexical_docs,
        }
    }

    pub fn embedding(&self) -> &CorpusEmbedding {
        &self.embedding
    }

    /// Documents ranked by relevance to `query`, strongest first — a hybrid of
    /// dense (semantic) and lexical (exact-term) retrieval fused by reciprocal
    /// rank. Lexical recall covers rare/out-of-vocabulary terms the embedding
    /// drops; either arm being empty degrades gracefully to the other. The score
    /// is the fusion score, not a cosine.
    pub fn search(&self, query: &str, limit: usize) -> Vec<(String, f64)> {
        let pool = (limit * 5).max(20);
        let dense: Vec<String> = self
            .dense_search(query, pool)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        let lexical: Vec<String> = self
            .lexical_docs
            .search(query, pool)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        crate::craft::lexical::reciprocal_rank_fusion(
            &[dense, lexical],
            crate::craft::lexical::RRF_K,
            limit,
        )
    }

    /// Documents ranked by dense cosine similarity to `query` alone.
    fn dense_search(&self, query: &str, limit: usize) -> Vec<(String, f64)> {
        let q = self.embedding.embed_text(query);
        let mut hits: Vec<(String, f64)> = self
            .doc_vectors
            .iter()
            .map(|(id, v)| {
                (
                    id.clone(),
                    crate::substrate::utils::cosine_similarity(&q, v),
                )
            })
            .collect();
        crate::substrate::utils::sort_by_score_desc(&mut hits);
        hits.truncate(limit);
        hits
    }

    /// Manuscript-tuned thesaurus for `word` (delegates to the embedding).
    pub fn related_words(&self, word: &str, k: usize) -> Vec<(String, f64)> {
        self.embedding.nearest_words(word, k)
    }

    /// Scenes ranked by similarity to `query`, strongest first, as
    /// `(scene, score)` — passage-level retrieval across the whole manuscript.
    pub fn search_scenes(&self, query: &str, limit: usize) -> Vec<(SceneRef, f64)> {
        self.scenes.search(&self.embedding, query, limit)
    }

    /// The scene index over the manuscript's shared embedding space.
    pub fn scene_index(&self) -> &SceneIndex {
        &self.scenes
    }

    /// Recompute one document's cached vector against the existing embedding
    /// model (vocabulary/SVD unchanged), adding it if new. Far cheaper than a
    /// full rebuild; the host periodically rebuilds so the model itself does not
    /// drift far from the evolving corpus.
    pub fn update_document(&mut self, id: &str, text: &str) {
        let vec = self.embedding.embed_text(text);
        match self.doc_vectors.iter_mut().find(|(d, _)| d == id) {
            Some(entry) => entry.1 = vec,
            None => self.doc_vectors.push((id.to_string(), vec)),
        }
        self.scenes.update_document(id, text, &self.embedding);
    }
}

/// Convert a co-occurrence matrix into a positive PMI matrix with
/// context-distribution smoothing (raise context marginals to `PPMI_SMOOTHING`),
/// which tempers the bias toward rare contexts and improves similarity quality.
fn ppmi_matrix(cooc: DMatrix<f64>) -> DMatrix<f64> {
    let n = cooc.nrows();
    let total: f64 = cooc.sum();
    if total <= 0.0 {
        return cooc;
    }
    let row_sums: Vec<f64> = (0..n).map(|i| cooc.row(i).sum()).collect();
    let col_smoothed: Vec<f64> = (0..n)
        .map(|j| cooc.column(j).sum().powf(PPMI_SMOOTHING))
        .collect();
    let col_total: f64 = col_smoothed.iter().sum();

    let mut ppmi = DMatrix::<f64>::zeros(n, n);
    if col_total <= 0.0 {
        return ppmi;
    }
    for i in 0..n {
        let p_w = row_sums[i] / total;
        if p_w <= 0.0 {
            continue;
        }
        for j in 0..n {
            let c = cooc[(i, j)];
            if c <= 0.0 {
                continue;
            }
            let p_wc = c / total;
            let p_c = col_smoothed[j] / col_total;
            let pmi = (p_wc / (p_w * p_c)).ln();
            if pmi > 0.0 {
                ppmi[(i, j)] = pmi;
            }
        }
    }
    ppmi
}

/// Truncated SVD of the PPMI matrix → dense word vectors `U_k · Σ_k^{1/2}`,
/// L2-normalized. Singular triplets are selected by magnitude (independent of
/// the solver's ordering).
fn svd_vectors(ppmi: &DMatrix<f64>, words: &[String], dim: usize) -> HashMap<String, Vec<f64>> {
    let n = ppmi.nrows();
    let (u, s) = truncated_svd(ppmi, dim);
    if u.ncols() == 0 {
        return HashMap::new();
    }

    // Rank singular values by magnitude and take the top `dim`.
    let mut order: Vec<usize> = (0..s.len()).collect();
    order.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal));
    order.truncate(dim);

    let mut vectors = HashMap::with_capacity(n);
    for (i, word) in words.iter().enumerate() {
        let mut v: Vec<f64> = order.iter().map(|&j| u[(i, j)] * s[j].sqrt()).collect();
        l2_normalize(&mut v);
        vectors.insert(word.clone(), v);
    }
    vectors
}

/// Number of singular triplets is small enough for an exact dense SVD at or
/// below this matrix size; larger matrices use randomized SVD.
const EXACT_SVD_MAX: usize = 200;

/// Left singular vectors and singular values of `m`, computed exactly for small
/// matrices and via deterministic randomized SVD (Halko et al.) for large ones,
/// so the dense PPMI factorization stays fast as the vocabulary grows. `m` is
/// symmetric (PPMI of a symmetric co-occurrence matrix), so power iterations use
/// `m` directly.
fn truncated_svd(m: &DMatrix<f64>, target: usize) -> (DMatrix<f64>, Vec<f64>) {
    let n = m.nrows();
    let l = (target + 10).min(n); // oversampled rank
    if n <= EXACT_SVD_MAX || l >= n {
        let svd = m.clone().svd(true, false);
        let u = svd.u.unwrap_or_else(|| DMatrix::zeros(n, 0));
        let s = svd.singular_values.iter().copied().collect();
        return (u, s);
    }

    // Randomized range finder with 2 power iterations.
    let omega = rademacher_projection(n, l);
    let mut y = m * &omega;
    for _ in 0..2 {
        let my = m * &y;
        y = m * &my;
    }
    let q = y.qr().q(); // n × l, orthonormal columns
    let b = q.transpose() * m; // l × n
    let svd_b = b.svd(true, false);
    let u_b = match svd_b.u {
        Some(u) => u,
        None => return (DMatrix::zeros(n, 0), Vec::new()),
    };
    let u = &q * &u_b; // n × l, approximate left singular vectors of m
    let s = svd_b.singular_values.iter().copied().collect();
    (u, s)
}

/// Deterministic ±1 (Rademacher) projection matrix. Entries are derived from
/// their indices via a splitmix64-style hash — no RNG — so the embedding is
/// reproducible across runs.
fn rademacher_projection(rows: usize, cols: usize) -> DMatrix<f64> {
    DMatrix::from_fn(rows, cols, |r, c| {
        let mut z = (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (c as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        if z & 1 == 0 {
            1.0
        } else {
            -1.0
        }
    })
}

fn l2_normalize(v: &mut [f64]) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A corpus with two clearly separated topics, repeated for signal.
    fn corpus() -> Vec<&'static str> {
        vec![
            "the king ruled the royal castle with the queen beside the throne",
            "the queen and king held royal court in the castle throne room",
            "the royal king crowned the queen in the castle throne hall",
            "the programmer wrote python code to fix the software bug today",
            "the software code compiled after the programmer fixed the python bug",
            "the python programmer debugged the code and shipped the software",
        ]
    }

    #[test]
    fn is_deterministic() {
        let texts = corpus();
        let a = CorpusEmbedding::train(&texts, &EmbeddingOptions::default());
        let b = CorpusEmbedding::train(&texts, &EmbeddingOptions::default());
        assert_eq!(a.word_vector("king"), b.word_vector("king"));
    }

    #[test]
    fn co_occurring_words_are_closer_than_unrelated() {
        let texts = corpus();
        let emb = CorpusEmbedding::train(&texts, &EmbeddingOptions::default());
        let king = emb.word_vector("king").expect("king in vocab");
        let queen = emb.word_vector("queen").expect("queen in vocab");
        let code = emb.word_vector("code").expect("code in vocab");
        let royal_sim = crate::substrate::utils::cosine_similarity(king, queen);
        let cross_sim = crate::substrate::utils::cosine_similarity(king, code);
        assert!(
            royal_sim > cross_sim,
            "king~queen ({royal_sim:.3}) should exceed king~code ({cross_sim:.3})"
        );
    }

    #[test]
    fn nearest_words_returns_same_topic() {
        let texts = corpus();
        let emb = CorpusEmbedding::train(&texts, &EmbeddingOptions::default());
        let near = emb.nearest_words("python", 3);
        assert!(!near.is_empty());
        let names: Vec<&str> = near.iter().map(|(w, _)| w.as_str()).collect();
        // The nearest neighbor should be from the programming topic, not royalty.
        assert!(
            names
                .iter()
                .any(|w| ["code", "software", "programmer", "bug"].contains(w)),
            "nearest to 'python': {names:?}"
        );
        assert!(!names.contains(&"throne"));
    }

    #[test]
    fn embed_text_similarity_tracks_topic() {
        let texts = corpus();
        let emb = CorpusEmbedding::train(&texts, &EmbeddingOptions::default());
        let same = emb.similarity("the king and queen", "royal castle throne");
        let diff = emb.similarity("the king and queen", "python software code");
        assert!(same > diff, "same-topic {same:.3} vs cross-topic {diff:.3}");
    }

    #[test]
    fn randomized_svd_path_handles_large_vocab() {
        // 260+ distinct tokens force the randomized SVD branch. "alpha" always
        // co-occurs with "beta" and never with "omega".
        let mut lines: Vec<String> = (0..260)
            .map(|i| format!("alpha beta tok{i} tok{i}"))
            .collect();
        for i in 0..5 {
            lines.push(format!("omega gamma delta extra{i} extra{i}"));
        }
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let emb = CorpusEmbedding::train(&refs, &EmbeddingOptions::default());
        assert!(
            emb.vocab_size() > super::EXACT_SVD_MAX,
            "vocab {} should exceed the exact-SVD threshold (randomized path)",
            emb.vocab_size()
        );

        let alpha = emb.word_vector("alpha").expect("alpha");
        let beta = emb.word_vector("beta").expect("beta");
        let omega = emb.word_vector("omega").expect("omega");
        let related = crate::substrate::utils::cosine_similarity(alpha, beta);
        let unrelated = crate::substrate::utils::cosine_similarity(alpha, omega);
        assert!(
            related > unrelated,
            "alpha~beta ({related:.3}) should exceed alpha~omega ({unrelated:.3})"
        );

        // Deterministic across runs despite the random projection.
        let emb2 = CorpusEmbedding::train(&refs, &EmbeddingOptions::default());
        assert_eq!(emb.word_vector("alpha"), emb2.word_vector("alpha"));
    }

    #[test]
    fn semantic_index_ranks_relevant_document_first() {
        let docs = vec![
            (
                "court".to_string(),
                "the royal king and royal queen sat on the throne in the castle \
                 the king and queen held court at the throne"
                    .to_string(),
            ),
            (
                "lab".to_string(),
                "the python programmer wrote python code the programmer fixed \
                 the code and the software bug software"
                    .to_string(),
            ),
        ];
        let index = SemanticIndex::build(&docs, &EmbeddingOptions::default());
        let hits = index.search("royal queen throne", 5);
        assert_eq!(hits[0].0, "court", "ranked: {hits:?}");
        // The thesaurus is reachable through the index too.
        assert!(!index.related_words("king", 3).is_empty());
    }

    #[test]
    fn update_document_reembeds_only_that_document() {
        let docs = vec![
            (
                "court".to_string(),
                "the royal king and royal queen sat on the throne in the castle".to_string(),
            ),
            (
                "lab".to_string(),
                "the python programmer wrote python code and fixed the software bug".to_string(),
            ),
        ];
        let mut index = SemanticIndex::build(&docs, &EmbeddingOptions::default());
        // Initially the royalty query ranks "court" first.
        assert_eq!(index.search("royal queen throne", 2)[0].0, "court");
        // Rewrite "lab" to royalty content; its cached vector updates in place.
        index.update_document(
            "lab",
            "the queen and king ruled the royal throne and castle court",
        );
        let hits = index.search("royal queen throne", 2);
        // Both documents are now royalty-relevant.
        assert!(hits.iter().all(|(_, s)| *s > 0.0), "hits: {hits:?}");
        assert!(hits.iter().any(|(id, _)| id == "lab"));
    }

    #[test]
    fn unknown_word_has_no_neighbors() {
        let texts = corpus();
        let emb = CorpusEmbedding::train(&texts, &EmbeddingOptions::default());
        assert!(emb.nearest_words("zzzznotaword", 5).is_empty());
    }

    #[test]
    fn weighted_pooling_concentrates_on_salient_rare_words() {
        // "walked" is frequent; "guillotine" is rare but salient. SIF pooling
        // should pull a passage vector toward the rare word more than the flat
        // mean it replaced would have.
        let texts = [
            "the soldier walked to the square and walked again",
            "the soldier walked past the guillotine in the square",
            "the crowd walked toward the guillotine in the square",
            "soldiers walked and walked across the square",
        ];
        let emb = CorpusEmbedding::train(&texts, &EmbeddingOptions::default());
        let salient = emb.word_vector("guillotine").expect("guillotine in vocab");

        let passage = "walked walked walked guillotine";
        let weighted = emb.embed_text(passage);

        // Flat mean of the same in-vocabulary words, the prior behavior.
        let mut mean = vec![0.0; emb.dim()];
        let mut n = 0usize;
        for w in ["walked", "walked", "walked", "guillotine"] {
            if let Some(v) = emb.word_vector(w) {
                for (s, x) in mean.iter_mut().zip(v) {
                    *s += x;
                }
                n += 1;
            }
        }
        for s in mean.iter_mut() {
            *s /= n as f64;
        }
        let norm = mean.iter().map(|x| x * x).sum::<f64>().sqrt();
        for s in mean.iter_mut() {
            *s /= norm;
        }

        let weighted_sim = crate::substrate::utils::cosine_similarity(&weighted, salient);
        let mean_sim = crate::substrate::utils::cosine_similarity(&mean, salient);
        assert!(
            weighted_sim > mean_sim,
            "weighted pooling should lean toward the salient rare word: \
             weighted={weighted_sim}, mean={mean_sim}"
        );
    }

    #[test]
    fn scene_search_finds_passage_across_documents() {
        // Two documents, each two scenes. A royalty query should surface the
        // royalty scene from whichever document it lives in, in the shared space.
        // Terms repeat so they clear min_count and land in the vocabulary
        // (single-occurrence words are dropped — the dense blind spot).
        let docs = vec![
            (
                "d1".to_string(),
                "the python code ran and the python code compiled\n\n\
                 the royal throne stood as the royal king held the royal throne"
                    .to_string(),
            ),
            (
                "d2".to_string(),
                "soldiers marched and soldiers marched across the field\n\n\
                 the queen and the royal queen ruled the royal court"
                    .to_string(),
            ),
        ];
        let index = SemanticIndex::build(&docs, &EmbeddingOptions::default());
        // Scenes from both documents share one index (cross-document).
        assert_eq!(index.scene_index().len(), 4);

        let hits = index.search_scenes("royal throne", 4);
        assert!(!hits.is_empty());
        let top = &hits[0].0;
        assert_eq!(top.document_id, "d1");
        assert_eq!(top.scene_index, 1, "the throne scene is the second block");
        assert!(top.utf16_end > top.utf16_start);
        assert!(top.snippet.contains("throne"), "snippet: {}", top.snippet);
    }

    #[test]
    fn scene_locators_count_utf16_code_units() {
        // A non-BMP char (🜂, 2 UTF-16 code units) before the second scene must
        // shift that scene's offset by 2, matching the editor's edit offsets.
        let docs = vec![(
            "d1".to_string(),
            "🜂 alpha beta\n\ngamma delta epsilon".to_string(),
        )];
        let index = SemanticIndex::build(&docs, &EmbeddingOptions::default());
        let scenes = index.scene_index().scenes();
        assert_eq!(scenes.len(), 2);
        // First scene "🜂 alpha beta" is 13 UTF-16 units (🜂=2, " alpha beta"=11);
        // the blank line "\n\n" adds 2, so the second scene starts at 15.
        assert_eq!(scenes[0].utf16_start, 0);
        assert_eq!(scenes[0].utf16_end, 13);
        assert_eq!(scenes[1].utf16_start, 15);
    }

    #[test]
    fn hybrid_search_finds_out_of_vocabulary_terms() {
        // A name occurring once is dropped from the embedding vocabulary, so
        // dense search has no signal for it; the lexical arm still finds it.
        let docs = vec![
            (
                "d1".to_string(),
                "the cat sat on the warm mat and the cat slept by the warm fire".to_string(),
            ),
            (
                "d2".to_string(),
                "a dog ran across the muddy field and the dog barked at dawn".to_string(),
            ),
            (
                "d3".to_string(),
                "the zarathustra manuscript lay hidden where the scholars once met".to_string(),
            ),
        ];
        let index = SemanticIndex::build(&docs, &EmbeddingOptions::default());
        // The query term really is out of the dense vocabulary.
        assert!(index.embedding().word_vector("zarathustra").is_none());
        let hits = index.search("zarathustra", 3);
        assert_eq!(
            hits[0].0, "d3",
            "hybrid should surface the exact-term doc: {hits:?}"
        );
    }

    #[test]
    fn redundant_pairs_surface_repeated_passages() {
        // The same passage in two documents should pair as redundant.
        let storm = "the storm broke over the harbor as waves crashed against the stone pier";
        let docs = vec![
            (
                "ch1".to_string(),
                format!("{storm}\n\nshe walked home through the quiet evening streets"),
            ),
            (
                "ch9".to_string(),
                format!("the council argued for hours about the harvest\n\n{storm}"),
            ),
        ];
        let index = SemanticIndex::build(&docs, &EmbeddingOptions::default());
        let pairs = index.scene_index().redundant_pairs(2, 5);
        assert!(!pairs.is_empty(), "expected a repeated-passage pair");
        let (a, b, score) = &pairs[0];
        assert_ne!(a.document_id, b.document_id);
        assert!(
            *score > 0.5,
            "near-duplicate scenes should score high: {score}"
        );
    }
}
