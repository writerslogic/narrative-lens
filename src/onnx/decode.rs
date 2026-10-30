//! Runtime-agnostic post-processing shared by both [`InferenceBackend`] impls
//! (`ort` natively, `tract` on wasm).
//!
//! The model runtimes differ, but everything *after* the raw tensor comes back
//! is pure arithmetic over `f32`/`i64` buffers: masked mean-pooling for SBERT
//! and BIO span decoding for NER. Keeping it here — with no `ort`/`tract`
//! dependency — means the two backends share exactly one implementation, so identical output tensors share the same decoding on both backends, and this logic
//! is unit-tested on the host gate even though the wasm backend cannot be.
//!
//! [`InferenceBackend`]: super::InferenceBackend

use super::model::NER_LABEL_NAMES;
use crate::substrate::utils::is_honorific;

/// Masked mean-pool a batch of token embeddings into one sentence vector per row.
///
/// `data` is the row-major `[batch, max_len, hidden_dim]` token-embedding tensor;
/// `row_masks[b]` is row `b`'s attention mask padded to `max_len`. Padding tokens
/// (mask 0) contribute zero weight and zero value, so a row's pooled vector is
/// identical to embedding that text with no padding at all. A row whose mask is
/// all-zero pools to an all-zero vector (it never divides by zero).
///
/// `batch` is `row_masks.len()`. Returns one `Vec<f64>` of length `hidden_dim`
/// per row, in order.
pub(super) fn masked_mean_pool(
    data: &[f32],
    row_masks: &[Vec<i64>],
    max_len: usize,
    hidden_dim: usize,
) -> Vec<Vec<f64>> {
    let mut results: Vec<Vec<f64>> = Vec::with_capacity(row_masks.len());
    for (b, mask) in row_masks.iter().enumerate() {
        let row_base = b * max_len * hidden_dim;
        let mut pooled = vec![0.0f64; hidden_dim];
        let mut total_weight = 0.0f64;
        for (i, &m) in mask.iter().enumerate() {
            let w = m as f64;
            if w == 0.0 {
                continue;
            }
            total_weight += w;
            let base = row_base + i * hidden_dim;
            for j in 0..hidden_dim {
                pooled[j] += data[base + j] as f64 * w;
            }
        }
        if total_weight > 0.0 {
            for v in &mut pooled {
                *v /= total_weight;
            }
        }
        results.push(pooled);
    }
    results
}

/// Numerically-stable softmax over one logit row → probabilities as `f64`.
///
/// Shifts by the row max before exponentiating so large logits do not overflow.
/// An empty row returns empty; a degenerate row whose exponentials underflow to
/// zero returns an all-zero vector rather than NaN.
pub(super) fn softmax(logits: &[f32]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = logits.iter().map(|&l| ((l - max) as f64).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|e| e / sum).collect()
    } else {
        vec![0.0; logits.len()]
    }
}

/// Argmax the per-token label for row `b` over its `seq_len` real tokens.
///
/// `logits` is the row-major `[batch, max_len, num_labels]` tensor; row `b`'s
/// token `i` starts at `(b * max_len + i) * num_labels`. Returns one label index
/// per real token (padding tokens beyond `seq_len` are not decoded). A supplied
/// canonical-to-source map makes both winners and ties use canonical indices.
pub(super) fn argmax_labels(
    logits: &[f32],
    b: usize,
    max_len: usize,
    num_labels: usize,
    seq_len: usize,
    canonical_to_source: Option<&[usize]>,
) -> Vec<usize> {
    let mut predicted: Vec<usize> = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        let base = (b * max_len + i) * num_labels;
        let mut best = (0usize, f32::NEG_INFINITY);
        for j in 0..num_labels {
            let source = canonical_to_source.map_or(j, |order| order[j]);
            if logits[base + source] > best.1 {
                best = (j, logits[base + source]);
            }
        }
        predicted.push(best.0);
    }
    predicted
}

/// The entity type of a BIO label: `"B-PER"`/`"I-PER"` → `"PER"`, `"O"` → `"O"`.
fn entity_type(label: &str) -> &str {
    if let Some(pos) = label.find('-') {
        &label[pos + 1..]
    } else {
        label
    }
}

/// Push `(span, type, start, end)` for a completed entity if its byte range is
/// in bounds and holds a letter or digit. Out-of-range, blank or punctuation-only
/// spans are dropped.
fn flush_entity(
    entities: &mut Vec<(String, String, usize, usize)>,
    text: &str,
    etype: &str,
    start: usize,
    end: usize,
) {
    if let Some(span) = text.get(start..end) {
        let trimmed = span.trim();
        let leading = span.len() - span.trim_start().len();
        if trimmed.chars().any(char::is_alphanumeric) {
            entities.push((
                trimmed.into(),
                etype.into(),
                start + leading,
                start + leading + trimmed.len(),
            ));
        }
    }
}

/// Group per-token BIO label predictions into `(text, type, start, end)` entity
/// spans over `text`. `offsets[i]` is token `i`'s `(start, end)` byte range in
/// `text`; a `(0, 0)` offset marks a special token (CLS/SEP/pad) and closes any
/// open entity. `words[i]` is the tokenizer's word id: a token repeating the
/// previous word id is a wordpiece continuation, which takes the word's first
/// token label (the only one the model is trained on) and only extends the
/// span, so a name split into pieces stays one entity. Shared by the single and
/// batched NER paths so there is exactly one decode implementation across both
/// backends.
pub(super) fn decode_entities(
    predicted: &[usize],
    offsets: &[(usize, usize)],
    words: &[Option<u32>],
    text: &str,
) -> Vec<(String, String, usize, usize)> {
    let mut entities = decode_bio(predicted, offsets, words, text, &NER_LABEL_NAMES);
    attach_person_titles(&mut entities, text);
    entities
}

/// BIO span grouping over any label set (`label_names[id]`, `"O"` outside);
/// see [`decode_entities`] for the token, offset, and wordpiece conventions.
pub(super) fn decode_bio(
    predicted: &[usize],
    offsets: &[(usize, usize)],
    words: &[Option<u32>],
    text: &str,
    label_names: &[&str],
) -> Vec<(String, String, usize, usize)> {
    let mut entities: Vec<(String, String, usize, usize)> = Vec::new();
    let mut cur_type: Option<&str> = None;
    let mut cur_start = 0usize;
    let mut cur_end = 0usize;
    let mut previous_word = None;

    for (i, &lid) in predicted.iter().enumerate() {
        if i >= offsets.len() {
            break;
        }
        if offsets[i].0 == offsets[i].1 {
            if let Some(et) = cur_type.take() {
                flush_entity(&mut entities, text, et, cur_start, cur_end);
            }
            previous_word = None;
            continue;
        }
        let word = words.get(i).copied().flatten();
        let continuation = word.is_some() && word == previous_word;
        previous_word = word;
        if continuation {
            if cur_type.is_some() {
                cur_end = offsets[i].1;
            }
            continue;
        }
        let label = label_names.get(lid).copied().unwrap_or("O");
        if label.starts_with("B-") {
            if let Some(et) = cur_type.take() {
                flush_entity(&mut entities, text, et, cur_start, cur_end);
            }
            cur_type = Some(entity_type(label));
            cur_start = offsets[i].0;
            cur_end = offsets[i].1;
        } else if label.starts_with("I-") {
            let it = entity_type(label);
            if cur_type == Some(it) {
                cur_end = offsets[i].1;
            } else {
                if let Some(et) = cur_type.take() {
                    flush_entity(&mut entities, text, et, cur_start, cur_end);
                }
                cur_type = Some(it);
                cur_start = offsets[i].0;
                cur_end = offsets[i].1;
            }
        } else if let Some(et) = cur_type.take() {
            flush_entity(&mut entities, text, et, cur_start, cur_end);
        }
    }
    if let Some(et) = cur_type {
        flush_entity(&mut entities, text, et, cur_start, cur_end);
    }
    entities
}

/// Start of a capitalised honorific that directly precedes `start` (`Mr. `,
/// `Mrs `), if any, not reaching before `floor`.
fn title_before(text: &str, start: usize, floor: usize) -> Option<usize> {
    let head = text.get(floor..start)?;
    let gap = head.trim_end_matches(['.', ' ']);
    let separator = &head[gap.len()..];
    if separator.chars().filter(|&ch| ch == '.').count() > 1
        || separator.chars().filter(|&ch| ch == ' ').count() > 1
        || separator.is_empty()
    {
        return None;
    }
    let word_start = gap
        .rfind(|ch: char| !ch.is_alphabetic())
        .map_or(0, |index| {
            index + gap[index..].chars().next().map_or(1, char::len_utf8)
        });
    let word = &gap[word_start..];
    (word.chars().next()?.is_uppercase() && is_honorific(word)).then_some(floor + word_start)
}

/// Trim leading punctuation from person spans (`. Todd` after an untagged
/// `Mrs`) and extend them over an adjacent honorific, so `Mrs. Todd` is one
/// entity as written. Never reaches into a preceding entity.
fn attach_person_titles(entities: &mut [(String, String, usize, usize)], text: &str) {
    let mut floor = 0;
    for entity in entities.iter_mut() {
        if entity.1 == "PER" {
            let skipped = text[entity.2..entity.3]
                .find(char::is_alphanumeric)
                .unwrap_or(0);
            let start = entity.2 + skipped;
            let start = title_before(text, start, floor).unwrap_or(if skipped > 0 {
                start
            } else {
                entity.2
            });
            if start != entity.2 {
                entity.2 = start;
                entity.0 = text[start..entity.3].trim().into();
            }
        }
        floor = entity.3;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_type_strips_bio_prefix() {
        assert_eq!(entity_type("B-PER"), "PER");
        assert_eq!(entity_type("I-LOC"), "LOC");
        assert_eq!(entity_type("O"), "O");
    }

    #[test]
    fn masked_mean_pool_ignores_padding() {
        // batch=1, max_len=3, hidden=2. Third token is padding (mask 0) with a
        // huge value that must not leak into the pooled result.
        let data = vec![
            1.0, 2.0, // token 0
            3.0, 4.0, // token 1
            99.0, 99.0, // token 2 (padding)
        ];
        let masks = vec![vec![1i64, 1, 0]];
        let pooled = masked_mean_pool(&data, &masks, 3, 2);
        assert_eq!(pooled.len(), 1);
        // mean of tokens 0 and 1 only: (1+3)/2=2, (2+4)/2=3
        assert!((pooled[0][0] - 2.0).abs() < 1e-12);
        assert!((pooled[0][1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn masked_mean_pool_all_padding_is_zero() {
        let data = vec![5.0, 6.0];
        let masks = vec![vec![0i64]];
        let pooled = masked_mean_pool(&data, &masks, 1, 2);
        assert_eq!(pooled[0], vec![0.0, 0.0]);
    }

    #[test]
    fn masked_mean_pool_two_rows_independent() {
        // batch=2, max_len=2, hidden=1. Row 0 pools token0; row1 pools both.
        let data = vec![10.0, 0.0, 2.0, 4.0];
        let masks = vec![vec![1i64, 0], vec![1i64, 1]];
        let pooled = masked_mean_pool(&data, &masks, 2, 1);
        assert!((pooled[0][0] - 10.0).abs() < 1e-12);
        assert!((pooled[1][0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn argmax_labels_picks_highest_logit() {
        // batch=1, max_len=2, num_labels=3.
        let logits = vec![
            0.1, 0.9, 0.2, // token 0 -> label 1
            5.0, 1.0, 2.0, // token 1 -> label 0
        ];
        let got = argmax_labels(&logits, 0, 2, 3, 2, None);
        assert_eq!(got, vec![1, 0]);
    }

    #[test]
    fn argmax_ties_use_the_requested_order_and_preserve_legacy_source_order() {
        let logits = [0.0, 0.0, 0.0, 2.0, 2.0, -1.0];
        assert_eq!(argmax_labels(&logits, 0, 2, 3, 2, None), [0, 0]);
        assert_eq!(argmax_labels(&logits, 0, 2, 3, 2, Some(&[2, 0, 1])), [0, 1]);
    }

    #[test]
    fn softmax_is_stable_and_normalized() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        assert_eq!(p.len(), 3);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p[2] > p[1] && p[1] > p[0]);
        // Large equal logits must not overflow to NaN; they split evenly.
        let big = softmax(&[1000.0, 1000.0]);
        assert!((big[0] - 0.5).abs() < 1e-9 && (big[1] - 0.5).abs() < 1e-9);
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn decode_entities_groups_bio_spans() {
        // "Alice Smith went" — tokens: CLS, Alice, Smith, went, SEP.
        let text = "Alice Smith went";
        // label ids: 3=B-PER, 4=I-PER, 0=O
        let predicted = vec![0usize, 3, 4, 0, 0];
        let offsets = vec![(0, 0), (0, 5), (6, 11), (12, 16), (0, 0)];
        let ents = decode_entities(&predicted, &offsets, &[], text);
        assert_eq!(ents.len(), 1);
        assert_eq!(ents[0].0, "Alice Smith");
        assert_eq!(ents[0].1, "PER");
        assert_eq!((ents[0].2, ents[0].3), (0, 11));
    }

    #[test]
    fn decode_entities_separate_b_starts_new_entity() {
        let text = "Paris London";
        let predicted = vec![7usize, 7]; // B-LOC, B-LOC
        let offsets = vec![(0, 5), (6, 12)];
        let ents = decode_entities(&predicted, &offsets, &[], text);
        assert_eq!(ents.len(), 2);
        assert_eq!(ents[0].0, "Paris");
        assert_eq!(ents[1].0, "London");
        assert!(ents.iter().all(|e| e.1 == "LOC"));
    }

    #[test]
    fn decoded_unicode_quotes_and_trimmed_ranges_agree() {
        let text = "  Zoë  ";
        let entities = decode_entities(&[3], &[(0, text.len())], &[], text);
        assert_eq!(entities.len(), 1);
        let (quote, label, start, end) = &entities[0];
        assert_eq!(quote, "Zoë");
        assert_eq!(label, "PER");
        assert_eq!(&text[*start..*end], quote);
        assert!(decode_entities(&[3], &[(3, 4)], &[], "Zoë").is_empty());
    }

    #[test]
    fn wordpiece_continuations_extend_the_first_piece_label() {
        // "Marilla smiled": Mari ##lla smiled; the model tags the pieces
        // B-PER, B-PER, then O, which must still be one PER span.
        let text = "Marilla smiled";
        let predicted = [3usize, 3, 0];
        let offsets = [(0, 4), (4, 7), (8, 14)];
        let words = [Some(0), Some(0), Some(1)];
        let ents = decode_entities(&predicted, &offsets, &words, text);
        assert_eq!(ents, [("Marilla".into(), "PER".into(), 0, 7)]);
        // A continuation never opens an entity when its word starts as O.
        let ents = decode_entities(&[0, 3, 0], &offsets, &words, text);
        assert!(ents.is_empty());
        // Adjacent distinct words stay separate: the apostrophe is its own word
        // and, being punctuation, never joins or becomes an entity.
        let text = "Rachel's";
        let offsets = [(0, 6), (6, 7), (7, 8)];
        let words = [Some(0), Some(1), Some(2)];
        let ents = decode_entities(&[3, 3, 0], &offsets, &words, text);
        assert_eq!(ents, [("Rachel".into(), "PER".into(), 0, 6)]);
    }

    #[test]
    fn person_spans_absorb_adjacent_honorifics_and_drop_punctuation_only_spans() {
        let text = "Mrs. Todd met Sir Henry and “ Dr Livesey .";
        let at = |needle: &str| {
            let start = text.find(needle).unwrap();
            (start, start + needle.len())
        };
        // `Todd` and `Henry Curtis`-style spans exclude their honorific; a
        // stray `.` span after an untagged `Mrs` starts mid-title.
        let offsets = [at(". Todd"), at("Henry"), at("“"), at("Livesey")];
        let ents = decode_entities(&[3, 3, 3, 3], &offsets, &[], text);
        let got: Vec<_> = ents.iter().map(|e| e.0.as_str()).collect();
        assert_eq!(got, ["Mrs. Todd", "Sir Henry", "Dr Livesey"]);
        for (quote, _, start, end) in &ents {
            assert_eq!(&text[*start..*end], quote);
        }
        // Non-person entities and ordinary words are left alone.
        let ents = decode_entities(&[7], &[at("Todd")], &[], text);
        assert_eq!(ents[0].0, "Todd");
        let ents = decode_entities(&[3], &[at("met")], &[], text);
        assert_eq!(ents[0].0, "met");
    }
}
