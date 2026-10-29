use narrative_lens::manuscript::{
    AnalysisContext, Configuration, Manuscript, Scene, analyze_manuscript, compare_reports,
};
fn main() -> narrative_lens::Result<()> {
    let mut manuscript = Manuscript {
        document_id: "novel".into(),
        scenes: vec![Scene::new("opening", "She was was tired.")],
        known_characters: vec![],
        configuration: Configuration::default(),
    };
    let before = analyze_manuscript(&manuscript)?;
    assert!(
        before
            .findings
            .iter()
            .any(|f| f.rule_id == "NL-GRAMMAR-REPETITION")
    );
    manuscript.scenes[0].text = "She was tired.".into();
    let mut context = AnalysisContext::default();
    let after = context.analyze(&manuscript)?;
    assert_eq!(compare_reports(&before, &after).resolved.len(), 1);
    Ok(())
}
