use narrative_lens::manuscript::*;
use std::time::Instant;
fn measure(action: impl FnOnce()) -> f64 {
    let start = Instant::now();
    action();
    start.elapsed().as_secs_f64() * 1000.0
}
fn main() {
    let seed: serde_json::Value =
        serde_json::from_str(include_str!("../tests/fixtures/manuscript-passages.json")).unwrap();
    let source = seed["scenes"].as_array().unwrap();
    let mut m = Manuscript {
        document_id: "benchmark-novel-v1".into(),
        scenes: (0..160)
            .map(|i| {
                Scene::new(
                    format!("scene-{i:03}"),
                    source[i % source.len()].as_str().unwrap().repeat(14),
                )
            })
            .collect(),
        known_characters: vec!["Maren".into(), "Adrian".into()],
        configuration: Configuration::default(),
    };
    if let Some(path) = std::env::args().nth(1) {
        m = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    }
    let original = m.clone();
    let words = m
        .scenes
        .iter()
        .map(|s| s.text.split_whitespace().count())
        .sum::<usize>();
    let bytes = m.scenes.iter().map(|s| s.text.len()).sum::<usize>();
    let mut context = AnalysisContext::default();
    let mut full = Vec::new();
    let mut incremental = Vec::new();
    let mut reorder = Vec::new();
    for _ in 0..5 {
        m = original.clone();
        context.clear();
        full.push(measure(|| {
            context.analyze(&m).unwrap();
        }));
        m.scenes[80].text.push_str(" Snow fell.");
        incremental.push(measure(|| {
            context.analyze(&m).unwrap();
        }));
        m.scenes.swap(40, 41);
        reorder.push(measure(|| {
            context.analyze(&m).unwrap();
        }));
    }
    let stats = context.stats();
    let report = context.analyze(&m).unwrap();
    assert_eq!(report, analyze_manuscript(&m).unwrap());
    println!("{}",serde_json::to_string_pretty(&serde_json::json!({"fixture":m.document_id,"profile":"release; default features; no ONNX models","iterations":5,"wordsAtStart":words,"utf8BytesAtStart":bytes,"scenes":m.scenes.len(),"fullMs":full,"singleSceneEditMs":incremental,"reorderMs":reorder,"cache":stats,"finalMatchesFresh":true,"memoryScope":"cache retained payload and dependency metadata estimate; use OS process peak RSS separately; excludes current input/report, allocator bookkeeping and runtime models","method":"Each iteration resets identical input before timing; full analysis clears scene retention but does not reset process-wide regex/dictionary initialization. Compilation excluded."})).unwrap());
}
