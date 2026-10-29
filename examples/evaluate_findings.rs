#[path = "support/finding_evaluation.rs"]
mod evaluation;
fn main() {
    let corpus: evaluation::Corpus =
        serde_json::from_str(include_str!("../tests/fixtures/evaluation-v2.json")).unwrap();
    let controlled = evaluation::evaluate(&corpus, true);
    let raw = evaluation::evaluate(&corpus, false);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({"controlled":controlled,"raw":raw}))
            .unwrap()
    );
}
