#![no_main]

use libfuzzer_sys::fuzz_target;
use narrative_lens::craft::{dialogue, grammar, readability};
use narrative_lens::substrate::text::{byte_to_utf16_offset, utf16_len};

fuzz_target!(|text: &str| {
    let result = readability::compute_readability(text);
    for score in [
        result.fkgl,
        result.gunning_fog,
        result.smog,
        result.ari,
        result.coleman_liau,
        result.flesch_ease,
        result.dale_chall,
        result.readability_variance,
    ] {
        assert!(score.is_finite());
    }
    assert!(result.per_paragraph_fkgl.iter().all(|s| s.is_finite()));

    let _ = grammar::check(text);
    for line in dialogue::extract_dialogue(text, 0, &[]) {
        assert!((0.0..=1.0).contains(&line.confidence));
    }

    assert_eq!(utf16_len(text) as usize, text.encode_utf16().count());
    for offset in [0, text.len() / 2, text.len(), text.len() + 1] {
        let expected = text.get(..offset).map(|s| s.encode_utf16().count() as u32);
        assert_eq!(byte_to_utf16_offset(text, offset), expected);
    }
});
