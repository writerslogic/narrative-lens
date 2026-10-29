#![no_main]

use libfuzzer_sys::fuzz_target;
use narrative_lens::continuity::{anachronism, ner, world_state};
use narrative_lens::craft::{dialogue, grammar, readability};
use narrative_lens::structure::foreshadow;
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
    let smog = readability::analyze_smog(text);
    if let Some(grade) = smog.grade {
        assert!(grade.is_finite());
        assert_eq!(smog.sampled_sentences.len(), 30);
        assert!(smog.eligible_sentence_count >= 30);
    } else {
        assert!(smog.sampled_sentences.is_empty());
        assert!(smog.eligible_sentence_count < 30);
    }
    let mut previous_end = 0;
    for sample in smog.sampled_sentences {
        assert!(sample.byte_start >= previous_end);
        assert!(sample.byte_start < sample.byte_end);
        assert!(text.get(sample.byte_start..sample.byte_end).is_some());
        previous_end = sample.byte_end;
    }

    let _ = grammar::check(text);
    let _ = ner::ner_extract_entities(text);
    let scenes = text
        .split('\n')
        .take(32)
        .map(|scene| scene.split_whitespace().map(str::to_owned).collect())
        .collect();
    let _ = foreshadow::analyze_foreshadowing(scenes, text.len() % 64);
    let _ = anachronism::analyze_scene(&[text.into()], i32::MAX, 0);
    let world_scenes: Vec<_> = text.split('\n').take(8).collect();
    let world = world_state::track_world_state(&world_scenes, &["Alice", "Bob"]);
    let current = world_state::track_world_state_current(
        &world_scenes,
        &["Alice", "Bob"],
        &world_state::WorldStateOptions {
            include_deltas: true,
            include_event_history: true,
        },
    );
    let current = current.unwrap();
    let mut facts = std::collections::HashSet::new();
    for delta in &current.deltas {
        for fact in &delta.invalidated_facts {
            assert!(facts.remove(fact));
        }
        for fact in &delta.new_facts {
            assert!(facts.insert(fact.clone()));
        }
        assert_eq!(facts, world.snapshots[delta.after_scene].facts);
    }
    assert_eq!(facts, current.facts);
    assert_eq!(world.continuity_score, current.continuity_score);
    for line in dialogue::extract_dialogue(text, 0, &[]) {
        assert!((0.0..=1.0).contains(&line.confidence));
    }

    assert_eq!(utf16_len(text) as usize, text.encode_utf16().count());
    for offset in [0, text.len() / 2, text.len(), text.len() + 1] {
        let expected = text.get(..offset).map(|s| s.encode_utf16().count() as u32);
        assert_eq!(byte_to_utf16_offset(text, offset), expected);
    }
});
