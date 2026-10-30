//! Window planning and output decoding for the narrative multi-task model.
//!
//! The model scores one `[CLS] … [SEP]` window at a time: token classification
//! for proper and nominal person mentions, an antecedent matrix over the mention
//! candidates inside the window, and a speaker matrix over the quotations inside
//! it. Everything here is runtime-agnostic arithmetic over the tokenizer's
//! offsets and the raw output buffers, so `ort` and `tract` share one
//! implementation and the host gate unit-tests it.

use super::checked::{
    ModelFailure, NARRATIVE_WINDOW_SPAN_LIMIT, NARRATIVE_WINDOW_TOKENS, NarrativeRequest,
    NarrativeWindow,
};
use super::decode::{decode_bio, decode_entities};
use super::model::{ModelMetadata, NOMINAL_LABEL_NAMES};

/// The whole request text tokenized once without special tokens.
pub(super) struct Tokens {
    pub ids: Vec<i64>,
    pub offsets: Vec<(usize, usize)>,
    pub words: Vec<Option<u32>>,
    pub cls: i64,
    pub sep: i64,
}

/// One window's token range plus the request mentions and quotes that fall
/// entirely inside it as `(request index, local start, local end)`, where local
/// positions count `[CLS]` as 0.
pub(super) struct WindowPlan {
    pub token_start: usize,
    pub token_end: usize,
    pub mentions: Vec<(usize, usize, usize)>,
    pub quotes: Vec<(usize, usize, usize)>,
}

/// Model inputs for one window; empty span lists carry a `[CLS]` placeholder
/// because the graph needs at least one index per span input.
pub(super) struct WindowInputs {
    pub ids: Vec<i64>,
    pub mention_starts: Vec<i64>,
    pub mention_ends: Vec<i64>,
    pub quote_starts: Vec<i64>,
    pub quote_ends: Vec<i64>,
}

pub(super) fn tokenize(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
) -> Result<Tokens, ModelFailure> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|_| ModelFailure::execution("tokenization_failed"))?;
    let offsets = encoding.get_offsets().to_vec();
    let monotone = offsets
        .windows(2)
        .all(|pair| pair[0].0 <= pair[1].0 && pair[0].1 <= pair[1].1);
    if !monotone
        || offsets.iter().any(|&(a, b)| {
            a > b || b > text.len() || !text.is_char_boundary(a) || !text.is_char_boundary(b)
        })
    {
        return Err(ModelFailure::incompatible("tokenizer_invalid_offsets"));
    }
    let special = |token: &str| {
        tokenizer
            .token_to_id(token)
            .map(i64::from)
            .ok_or_else(|| ModelFailure::incompatible("model_profile_special_tokens"))
    };
    Ok(Tokens {
        ids: encoding.get_ids().iter().map(|&id| i64::from(id)).collect(),
        offsets,
        words: encoding.get_word_ids().to_vec(),
        cls: special("[CLS]")?,
        sep: special("[SEP]")?,
    })
}

/// Inclusive token span covering byte span `start..end`: the first token ending
/// after `start` through the last token starting before `end`. `None` when no
/// token overlaps the span.
pub(super) fn token_span(
    offsets: &[(usize, usize)],
    start: usize,
    end: usize,
) -> Option<(usize, usize)> {
    let first = offsets.partition_point(|&(_, token_end)| token_end <= start);
    let after_last = offsets.partition_point(|&(token_start, _)| token_start < end);
    (first < after_last).then(|| (first, after_last - 1))
}

pub(super) fn plan_windows(
    tokens: &Tokens,
    request: &NarrativeRequest<'_>,
) -> Result<Vec<WindowPlan>, ModelFailure> {
    let spans = |byte_spans: &[(usize, usize)]| -> Vec<Option<(usize, usize)>> {
        byte_spans
            .iter()
            .map(|&(start, end)| token_span(&tokens.offsets, start, end))
            .collect()
    };
    let mentions = spans(request.mentions);
    let quotes = spans(request.quotes);
    let inside = |spans: &[Option<(usize, usize)>], a: usize, b: usize| {
        let local: Vec<_> = spans
            .iter()
            .enumerate()
            .filter_map(|(index, span)| {
                span.filter(|&(first, last)| first >= a && last < b)
                    .map(|(first, last)| (index, first - a + 1, last - a + 1))
            })
            .collect();
        (local.len() <= NARRATIVE_WINDOW_SPAN_LIMIT)
            .then_some(local)
            .ok_or_else(|| ModelFailure::execution("narrative_window_span_limit"))
    };
    let total = tokens.ids.len();
    let mut plans = Vec::with_capacity(total.div_ceil(NARRATIVE_WINDOW_TOKENS));
    let mut a = 0usize;
    while a < total {
        let b = (a + NARRATIVE_WINDOW_TOKENS).min(total);
        plans.push(WindowPlan {
            token_start: a,
            token_end: b,
            mentions: inside(&mentions, a, b)?,
            quotes: inside(&quotes, a, b)?,
        });
        a = b;
    }
    Ok(plans)
}

pub(super) fn window_inputs(tokens: &Tokens, plan: &WindowPlan) -> WindowInputs {
    let mut ids = Vec::with_capacity(plan.token_end - plan.token_start + 2);
    ids.push(tokens.cls);
    ids.extend_from_slice(&tokens.ids[plan.token_start..plan.token_end]);
    ids.push(tokens.sep);
    let positions = |spans: &[(usize, usize, usize)], pick: fn(&(usize, usize, usize)) -> usize| {
        if spans.is_empty() {
            vec![0]
        } else {
            spans.iter().map(|span| pick(span) as i64).collect()
        }
    };
    WindowInputs {
        ids,
        mention_starts: positions(&plan.mentions, |span| span.1),
        mention_ends: positions(&plan.mentions, |span| span.2),
        quote_starts: positions(&plan.quotes, |span| span.1),
        quote_ends: positions(&plan.quotes, |span| span.2),
    }
}

/// Raw output buffers of one window run, already shape-checked by the backend:
/// `entity` is `[1, L, 9]`, `nominal` `[1, L, 3]`, `antecedent` `[m', m']`, and
/// `speaker` `[q', m' + 1]`, where `m'`/`q'` count the placeholder when the
/// window has no mentions/quotes.
pub(super) struct WindowOutputs<'a> {
    pub entity: &'a [f32],
    pub nominal: &'a [f32],
    pub antecedent: &'a [f32],
    pub speaker: &'a [f32],
}

pub(super) fn decode_window(
    text: &str,
    tokens: &Tokens,
    plan: &WindowPlan,
    metadata: &ModelMetadata,
    outputs: WindowOutputs<'_>,
) -> NarrativeWindow {
    let (a, b) = (plan.token_start, plan.token_end);
    let length = b - a + 2;
    let mut offsets = Vec::with_capacity(length);
    offsets.push((0, 0));
    offsets.extend_from_slice(&tokens.offsets[a..b]);
    offsets.push((0, 0));
    let mut words = Vec::with_capacity(length);
    words.push(None);
    words.extend_from_slice(&tokens.words[a..b]);
    words.push(None);
    let entity_labels = metadata.ner_labels(outputs.entity, 0, length, 9, length);
    let nominal_labels = metadata.nominal_labels(outputs.nominal, 0, length, length);
    let nominals = decode_bio(
        &nominal_labels,
        &offsets,
        &words,
        text,
        &NOMINAL_LABEL_NAMES,
    )
    .into_iter()
    .map(|(_, _, start, end)| (start, end))
    .collect();
    let m = plan.mentions.len();
    let antecedents = (0..m)
        .map(|i| {
            let mut row = Vec::with_capacity(m + 1);
            row.push(0.0);
            row.extend_from_slice(&outputs.antecedent[i * m..(i + 1) * m]);
            row
        })
        .collect();
    let stride = m.max(1) + 1;
    let speakers = (0..plan.quotes.len())
        .map(|q| outputs.speaker[q * stride..q * stride + m + 1].to_vec())
        .collect();
    NarrativeWindow {
        start: tokens.offsets[a].0,
        end: tokens.offsets[b - 1].1,
        entities: decode_entities(&entity_labels, &offsets, &words, text),
        nominals,
        mentions: plan.mentions.iter().map(|span| span.0).collect(),
        quotes: plan.quotes.iter().map(|span| span.0).collect(),
        antecedents,
        speakers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::checked::ModelKind;

    fn tokens(text: &str) -> Tokens {
        // One token per whitespace-separated word, in order.
        let mut offsets = Vec::new();
        let mut start = None;
        for (index, ch) in text.char_indices().chain([(text.len(), ' ')]) {
            match (ch.is_whitespace(), start) {
                (false, None) => start = Some(index),
                (true, Some(s)) => {
                    offsets.push((s, index));
                    start = None;
                }
                _ => {}
            }
        }
        Tokens {
            ids: (0..offsets.len() as i64).collect(),
            words: (0..offsets.len() as u32).map(Some).collect(),
            offsets,
            cls: 101,
            sep: 102,
        }
    }

    #[test]
    fn token_spans_cover_partial_and_multi_token_byte_ranges() {
        let offsets = tokens("Alice met her mother").offsets;
        assert_eq!(token_span(&offsets, 0, 5), Some((0, 0)));
        assert_eq!(token_span(&offsets, 10, 20), Some((2, 3)));
        assert_eq!(token_span(&offsets, 1, 3), Some((0, 0)));
        assert_eq!(token_span(&offsets, 5, 6), None);
        assert_eq!(token_span(&offsets, 0, 20), Some((0, 3)));
    }

    #[test]
    fn windows_cut_at_the_training_length_and_keep_only_enclosed_spans() {
        let text: String = (0..NARRATIVE_WINDOW_TOKENS + 3)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let tokens = tokens(&text);
        let word = |i: usize| tokens.offsets[i];
        let straddling = (
            word(NARRATIVE_WINDOW_TOKENS - 1).0,
            word(NARRATIVE_WINDOW_TOKENS).1,
        );
        let mentions = [
            word(0),
            word(5),
            straddling,
            word(NARRATIVE_WINDOW_TOKENS + 1),
        ];
        let quotes = [(word(1).0, word(3).1)];
        let request = NarrativeRequest {
            text: &text,
            mentions: &mentions,
            quotes: &quotes,
        };
        let plans = plan_windows(&tokens, &request).unwrap();
        assert_eq!(plans.len(), 2);
        assert_eq!(
            (plans[0].token_start, plans[0].token_end),
            (0, NARRATIVE_WINDOW_TOKENS)
        );
        assert_eq!(plans[0].mentions, vec![(0, 1, 1), (1, 6, 6)]);
        assert_eq!(plans[0].quotes, vec![(0, 2, 4)]);
        assert_eq!(plans[1].mentions, vec![(3, 2, 2)]);
        assert!(plans[1].quotes.is_empty());
        let inputs = window_inputs(&tokens, &plans[1]);
        assert_eq!(inputs.ids, vec![101, 400, 401, 402, 102]);
        assert_eq!(inputs.mention_starts, vec![2]);
        assert_eq!(inputs.quote_starts, vec![0]);
        assert_eq!(inputs.quote_ends, vec![0]);
    }

    #[test]
    fn decoding_maps_heads_back_to_bytes_and_prepends_the_no_antecedent_column() {
        let text = "Alice told her mother";
        let tokens = tokens(text);
        let mentions = [(0, 5), (11, 14), (11, 21)];
        let request = NarrativeRequest {
            text,
            mentions: &mentions,
            quotes: &[(6, 21)],
        };
        let plans = plan_windows(&tokens, &request).unwrap();
        let metadata = ModelMetadata::unverified(ModelKind::NarrativeMultitask);
        // Legacy source order is the canonical order: B-PER = 3, B-NOM = 1, I-NOM = 2.
        let one_hot = |labels: &[usize], width: usize| -> Vec<f32> {
            labels
                .iter()
                .flat_map(|&label| (0..width).map(move |j| if j == label { 1.0 } else { 0.0 }))
                .collect()
        };
        let entity = one_hot(&[0, 3, 0, 0, 0, 0], 9);
        let nominal = one_hot(&[0, 0, 0, 1, 2, 0], 3);
        let antecedent = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.5, 0.0, 0.0];
        let speaker = vec![0.1, 3.0, 0.2, 0.3];
        let window = decode_window(
            text,
            &tokens,
            &plans[0],
            &metadata,
            WindowOutputs {
                entity: &entity,
                nominal: &nominal,
                antecedent: &antecedent,
                speaker: &speaker,
            },
        );
        assert_eq!((window.start, window.end), (0, text.len()));
        assert_eq!(window.entities, vec![("Alice".into(), "PER".into(), 0, 5)]);
        assert_eq!(window.nominals, vec![(11, 21)]);
        assert_eq!(window.mentions, vec![0, 1, 2]);
        assert_eq!(window.quotes, vec![0]);
        assert_eq!(window.antecedents[1], vec![0.0, 2.0, 0.0, 0.0]);
        assert_eq!(window.speakers, vec![vec![0.1, 3.0, 0.2, 0.3]]);
    }

    #[test]
    fn placeholder_outputs_are_dropped_when_a_window_has_no_mentions() {
        let text = "Nobody spoke";
        let tokens = tokens(text);
        let request = NarrativeRequest {
            text,
            mentions: &[],
            quotes: &[(0, 12)],
        };
        let plans = plan_windows(&tokens, &request).unwrap();
        let inputs = window_inputs(&tokens, &plans[0]);
        assert_eq!(inputs.mention_starts, vec![0]);
        let metadata = ModelMetadata::unverified(ModelKind::NarrativeMultitask);
        let window = decode_window(
            text,
            &tokens,
            &plans[0],
            &metadata,
            WindowOutputs {
                entity: &[0.0; 4 * 9],
                nominal: &[0.0; 4 * 3],
                antecedent: &[0.0],
                speaker: &[0.7, 0.1],
            },
        );
        assert!(window.antecedents.is_empty());
        assert_eq!(window.speakers, vec![vec![0.7]]);
    }
}
