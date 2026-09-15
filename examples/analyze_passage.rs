//! Minimal end-to-end example: run a few analyzers over one passage.
//!
//! ```sh
//! cargo run --example analyze_passage
//! ```

use narrative_lens::craft::{grammar, readability};
use narrative_lens::structure::structure;

fn main() {
    let passage = "Maren stood at the edge of the dock, watching the last ferry \
        pull away without her. The water slapped against the pilings, cold and \
        indifferent. She had missed it on purpose, though she was not yet ready \
        to admit that to herself.";

    let readability = readability::compute_readability(passage);
    println!(
        "readability: fkgl={:.1} ({} words, {} sentences)",
        readability.fkgl, readability.word_count, readability.sentence_count
    );

    let findings = grammar::check(passage);
    println!("grammar: {} finding(s)", findings.len());

    let templates = structure::list_templates();
    println!("available structure templates: {}", templates.join(", "));
}
