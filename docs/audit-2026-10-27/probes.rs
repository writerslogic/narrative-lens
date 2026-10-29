use narrative_lens::{continuity::ner, substrate::text};

fn main() {
    for word in ["mistake", "explore", "secure", "marine", "sunrise"] {
        println!("syllables {word}: {}", text::count_syllables(word));
    }
    let s = "She waited. Then she left";
    println!("sentences count={} split={}", text::count_sentences(s), text::split_sentences(s).len());
    let mentions = ["John Smith", "Smith", "Mary Smith"].map(|s| narrative_lens::continuity::entity::Mention::new(s, vec![]));
    println!("surname bridge: {:?}", narrative_lens::continuity::entity::resolve(&mentions, narrative_lens::continuity::entity::EntityType::Character));
    println!("foreshadow mismatch: {:?}", std::panic::catch_unwind(|| narrative_lens::structure::foreshadow::analyze_foreshadowing(vec![], 3)));
    println!("anachronism max year: {:?}", std::panic::catch_unwind(|| narrative_lens::continuity::anachronism::analyze_scene(&["radio".to_owned()], i32::MAX, 0)));
    for whitespace in ['\u{0085}', '\u{00a0}', '\u{2003}', '\u{3000}'] {
        let input = format!("{}{}{}", "a".repeat(1490), whitespace, "b".repeat(20));
        let result = std::panic::catch_unwind(|| ner::ner_extract_entities(&input));
        println!("ner U+{:04X}: {}", whitespace as u32, if result.is_err() { "PANIC" } else { "ok" });
    }
}
