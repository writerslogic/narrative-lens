use narrative_lens::{
    craft::readability::compute_readability,
    substrate::{text::tokenize_words, word_stats::*},
};

#[test]
fn typographic_contractions_share_canonical_tokens_and_frequency_populations() {
    let straight = "Don't don't ‘rain’ RAIN O'Brien O’Brien";
    let tokens = tokenize_words(straight);
    assert_eq!(
        tokens,
        vec!["don't", "don't", "rain", "rain", "o'brien", "o'brien"]
    );
    let expected = word_frequencies_from_words(tokens, 10, |_| false);
    assert_eq!(word_frequencies(straight, 10, |_| false), expected);
    assert!(expected.iter().all(|word| word.count == 2));
    assert_eq!(
        word_frequencies_from_words(vec!["DON’T".into(), "don't".into()], 10, |_| false),
        vec![WordCount {
            word: "don't".into(),
            count: 2
        }]
    );
}

#[test]
fn quotation_edges_do_not_turn_familiar_words_into_difficult_words() {
    let plain = compute_readability("The cat sat.");
    for text in ["'The' cat sat.", "‘The’ cat sat."] {
        let quoted = compute_readability(text);
        assert_eq!(quoted.word_count, plain.word_count);
        assert_eq!(quoted.char_count, plain.char_count);
        assert_eq!(quoted.dale_chall, plain.dale_chall);
    }
}

#[test]
fn readability_formulas_do_not_depend_on_the_apostrophe_glyph() {
    let plain = compute_readability("The cat said, 'It isn't here.'");
    let curly = compute_readability("The cat said, ‘It isn’t here.’");
    assert_eq!(plain.word_count, curly.word_count);
    assert_eq!(plain.syllable_count, curly.syllable_count);
    assert_eq!(plain.char_count, curly.char_count);
    assert_eq!(plain.fkgl, curly.fkgl);
    assert_eq!(plain.dale_chall, curly.dale_chall);
}
