#[path = "support/evaluation.rs"]
mod evaluation;
fn main() {
    let corpus: evaluation::Corpus =
        serde_json::from_str(include_str!("../tests/fixtures/evaluation.json")).unwrap();
    if std::env::args().any(|a| a == "--legacy") {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({"corpusVersion":corpus.version,"annotationOrigin":corpus.annotation_origin,"mode":"legacy","metrics":evaluation::evaluate(&corpus,evaluation::legacy)})).unwrap());
        return;
    }
    let calibration = evaluation::fit_calibration(&corpus);
    let raw = evaluation::evaluate(&corpus, evaluation::facade);
    let calibrated = evaluation::evaluate(&corpus, |c| {
        if evaluation::facade(c) > 0.0 {
            calibration[&c.analyzer]
        } else {
            0.0
        }
    });
    println!("{}",serde_json::to_string_pretty(&serde_json::json!({"corpusVersion":corpus.version,"annotationOrigin":corpus.annotation_origin,"mode":"facade","metrics":raw,"calibration":{"method":"Laplace-smoothed emitted-case precision; fitted only on calibration; diagnostic only, not library defaults","scores":calibration},"calibratedMetrics":calibrated})).unwrap());
}
