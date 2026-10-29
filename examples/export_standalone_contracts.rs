use narrative_lens::{continuity::*, craft::*, structure::*, substrate::*};
use schemars::JsonSchema;
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    path::Path,
};
use ts_rs::{Config, TS, TypeVisitor};

#[derive(Default)]
struct Contracts {
    config: Config,
    visited: BTreeSet<String>,
    definitions: BTreeMap<String, String>,
    aliases: BTreeMap<String, String>,
    schemas: serde_json::Map<String, serde_json::Value>,
}
impl TypeVisitor for Contracts {
    fn visit<T: TS + 'static + ?Sized>(&mut self) {
        if !self.visited.insert(std::any::type_name::<T>().into()) {
            return;
        }
        if T::output_path().is_some() {
            let name = T::name(&self.config);
            let declaration = format!("export {}\n", T::decl(&self.config));
            if let Some(existing) = self.definitions.insert(name, declaration.clone()) {
                assert_eq!(existing, declaration, "conflicting standalone type names");
            }
        }
        T::visit_generics(self);
        T::visit_dependencies(self);
    }
}
impl Contracts {
    fn add<T: TS + JsonSchema + Serialize + 'static>(&mut self, alias: &str, _: T) {
        self.visit::<T>();
        self.aliases.insert(
            alias.into(),
            format!("export type {alias} = {};\n", T::name(&self.config)),
        );
        self.schemas.insert(
            alias.into(),
            serde_json::to_value(schemars::schema_for!(T)).unwrap(),
        );
    }
}
fn main() {
    let check = std::env::args().any(|s| s == "--check");
    let mut contracts = Contracts {
        config: Config::default().with_large_int("number"),
        ..Contracts::default()
    };
    macro_rules! emit {
        ($name:ident, $value:expr) => {
            contracts.add(stringify!($name), $value);
        };
    }
    emit!(
        AnalyzePromiseEventsResult,
        author_events::analyze_promise_events(
            &narrative_lens::manuscript::Manuscript {
                document_id: "contracts".into(),
                scenes: vec![],
                known_characters: vec![],
                configuration: Default::default(),
            },
            &[],
        )
        .unwrap()
    );
    contracts.add(
        "AuthorPromiseEvents",
        Vec::<author_events::AuthorPromiseEvent>::new(),
    );
    let author_manuscript = narrative_lens::manuscript::Manuscript {
        document_id: "contracts".into(),
        scenes: vec![],
        known_characters: vec![],
        configuration: Default::default(),
    };
    contracts.add(
        "ExtractPersonsResult",
        ner::ner_extract_persons_checked(&[]).unwrap(),
    );
    contracts.add(
        "GenreEvidenceContract",
        genre_compliance::GenreEvidence::default(),
    );
    contracts.add(
        "SuppliedUncertaintyObservations",
        Vec::<narrative_entropy::SuppliedUncertainty>::new(),
    );
    contracts.add(
        "SemanticNoveltyOptionsContract",
        narrative_entropy::SemanticNoveltyOptions::default(),
    );
    emit!(
        MeasureNarrativeUncertaintyResult,
        narrative_entropy::measure_narrative_uncertainty(&author_manuscript, &[]).unwrap()
    );
    emit!(
        MeasureSemanticNoveltyResult,
        narrative_entropy::measure_semantic_novelty(
            &[],
            &narrative_entropy::SemanticNoveltyOptions::default()
        )
        .unwrap()
    );
    emit!(
        AnalyzeAuthorKnowledgeResult,
        author_knowledge::analyze_author_knowledge(&author_manuscript, &[], &[]).unwrap()
    );
    contracts.add(
        "AuthorPropositions",
        Vec::<author_knowledge::AuthorProposition>::new(),
    );
    contracts.add(
        "AuthorKnowledgeEvents",
        Vec::<author_knowledge::AuthorKnowledgeEvent>::new(),
    );
    contracts.add(
        "AuthorEpistemicQueries",
        Vec::<epistemic_engine::AuthorEpistemicQuery>::new(),
    );
    emit!(
        AnalyzeAuthorEpistemicsResult,
        epistemic_engine::analyze_author_epistemics(&author_manuscript, &[], &[], &[]).unwrap()
    );
    contracts.add(
        "WorldStateOptionsContract",
        world_state::WorldStateOptions::default(),
    );
    emit!(
        TrackWorldStateCurrentResult,
        world_state::track_world_state_current(
            &[],
            &[],
            &world_state::WorldStateOptions::default()
        )
        .unwrap()
    );
    emit!(AnalyzeSmogResult, readability::analyze_smog(""));
    emit!(
        ExtractDialogueCheckedResult,
        dialogue::extract_dialogue_checked(
            "",
            0,
            &[],
            &dialogue::DialogueExtractionOptions::default()
        )
        .unwrap()
    );
    contracts.add(
        "DialogueExtractionOptionsContract",
        dialogue::DialogueExtractionOptions::default(),
    );
    contracts.add(
        "AuthorCausalNodes",
        Vec::<causal_engine::AuthorCausalNode>::new(),
    );
    contracts.add(
        "AuthorInterventions",
        Vec::<causal_engine::AuthorIntervention>::new(),
    );
    emit!(
        AnalyzeAuthorCausalityResult,
        causal_engine::analyze_author_causality(&author_manuscript, &[], &[]).unwrap()
    );
    emit!(
        BuildAuthorMentalModelResult,
        mental_model::build_author_mental_model(&author_manuscript, &[], &[]).unwrap()
    );
    contracts.add(
        "AuthorCoreferenceLinks",
        Vec::<coreference::AuthorCoreferenceLink>::new(),
    );
    emit!(
        ResolveAuthorCoreferenceResult,
        coreference::resolve_author_coreference(&author_manuscript, &[]).unwrap()
    );
    contracts.add(
        "AuthorSubtextAnnotations",
        Vec::<subtext::AuthorSubtextAnnotation>::new(),
    );
    emit!(
        AnalyzeSubtextAnnotationsResult,
        subtext::analyze_subtext_annotations(&author_manuscript, &[]).unwrap()
    );
    contracts.add(
        "AuthorThematicTheses",
        Vec::<thematic_argument::AuthorThematicThesis>::new(),
    );
    contracts.add(
        "AuthorThematicAnnotations",
        Vec::<thematic_argument::AuthorThematicAnnotation>::new(),
    );
    emit!(
        AnalyzeThematicAnnotationsResult,
        thematic_argument::analyze_thematic_annotations(&author_manuscript, &[], &[]).unwrap()
    );
    emit!(
        BuildVoiceProfileCheckedResult,
        voice::build_voice_profile_checked("contracts", &[]).unwrap()
    );
    emit!(AnalyzeSceneResult, anachronism::analyze_scene(&[], 0, 0));
    emit!(
        AnalyzeSceneCheckedResult,
        anachronism::analyze_scene_checked(&[], 0, 0).unwrap()
    );
    contracts.add(
        "HistoricalEntries",
        Vec::<anachronism::HistoricalEntry>::new(),
    );
    contracts.add(
        "HistoricalUsages",
        Vec::<anachronism::HistoricalUsage>::new(),
    );
    emit!(
        AnalyzeHistoricalCatalogueResult,
        anachronism::analyze_historical_catalogue(&author_manuscript, &[], &[], 0, 0).unwrap()
    );
    emit!(
        ExtractNamedEntitiesResult,
        ner::ner_extract_entities_batch_checked(&[]).unwrap()
    );
    emit!(
        MeasureFindingConfidenceResult,
        confidence::measure_finding_confidence(0, 0, 0.0).unwrap()
    );
    emit!(
        AnalyzeCausalityResult,
        causal_engine::analyze_causality(&[], &[])
    );
    emit!(
        ResolveEntitiesResult,
        entity::resolve(&[], entity::EntityType::Character)
    );
    emit!(
        ResolveEntitiesWithCorrectionsResult,
        entity::resolve_with_corrections(&[], entity::EntityType::Character, &[])
    );
    emit!(
        BuildMentalModelResult,
        mental_model::build_mental_model(&[], &[], &[], &[])
    );
    emit!(
        AnalyzeGapsResult,
        gap_analysis::analyze_gaps(&[], &[], &[], &[], 0, 0)
    );
    emit!(
        TrackEpistemicsResult,
        epistemic_engine::track_epistemics(&[], &[], &[])
    );
    emit!(BuildTimelineResult, timeline::build_timeline(&[]));
    contracts.add(
        "TimelineOverrides",
        Vec::<timeline::TimelineOverride>::new(),
    );
    emit!(
        TrackWorldStateResult,
        world_state::track_world_state(&[], &[])
    );
    emit!(
        ResolveCoreferencesResult,
        coreference::CoreferenceResolver::new().resolve("", &[], Some(&HashMap::new()))
    );
    emit!(AnalyzePacingResult, pacing::analyze_scene(0, 0, 0, 0));
    emit!(AggregatePacingResult, pacing::aggregate_pacing(&[]));
    emit!(
        ExtractDialogueResult,
        dialogue::extract_dialogue("", 0, &[])
    );
    emit!(
        AnalyzeDialoguePatternsResult,
        dialogue::analyze_dialogue_patterns(&[], &[])
    );
    emit!(
        AnalyzeDialogueRealismResult,
        dialogue_realism::analyze_dialogue_realism(&[], &[])
    );
    emit!(
        AnalyzeSyntaxTensionResult,
        syntax_tension::analyze_scene_text("")
    );
    emit!(
        GetTensionMetricsResult,
        syntax_tension::get_tension_metrics(&[])
    );
    emit!(
        CollectProseExamplesResult,
        prose_quality::collect_prose_examples(&[])
    );
    let voice_profile = voice::build_voice_profile("", &[]);
    emit!(BuildVoiceProfileResult, voice_profile.clone());
    emit!(CompareVoicesResult, voice::compare_voices(&[]));
    emit!(
        MeasureVoiceDriftResult,
        voice::measure_voice_drift(&voice_profile, &voice_profile)
    );
    emit!(
        MeasureVoiceDriftFromTextResult,
        voice::measure_voice_drift_from_text("", &[], &[])
    );
    emit!(
        AnalyzeCharacterVoicesResult,
        voice::analyze_character_voices(vec![])
    );
    emit!(
        AnalyzeSubtextResult,
        subtext::analyze_subtext(&[], &[], &[], "general")
    );
    emit!(
        GetStructureTemplateResult,
        structure::get_structure_template("three_act").unwrap()
    );
    emit!(
        ValidateStructureResult,
        structure::validate_structure("three_act", vec![], vec![]).unwrap()
    );
    emit!(
        AnalyzeForeshadowingResult,
        foreshadow::analyze_foreshadowing(vec![], 0)
    );
    emit!(
        BuildPromisePayoffLedgerResult,
        promise_payoff::build_promise_payoff_ledger(&[], &[], "general", &[], &[], "three_act", 0)
    );
    emit!(
        TrackThematicArgumentResult,
        thematic_argument::track_thematic_argument(&[], &[], &[], &[], "general")
    );
    emit!(
        AnalyzeOpeningResult,
        opening_analysis::analyze_opening("", "general", None)
    );
    emit!(
        CheckGenreComplianceResult,
        genre_compliance::check_genre_compliance(
            "general", false, false, false, true, false, false, 0, 0.0, ""
        )
    );
    emit!(
        ComputeNarrativeEntropyResult,
        narrative_entropy::compute_narrative_entropy(&[], &[], "three_act", "general", &[])
    );
    emit!(GenreFromLabelResult, genre::genre_from_label("general"));
    emit!(
        DetectGenreResult,
        genre::detect_genre(vec![], vec![], 0.0, 0.0)
    );
    emit!(
        DetectGenreSmartResult,
        genre::detect_genre_smart("", vec![], vec![], 0.0, 0.0)
    );
    emit!(
        EvaluateAgainstGenreResult,
        genre::evaluate_against_genre(genre::Genre::General, 0.0, 0.0, 0.0, 0.0, 0, 0)
    );
    emit!(
        GetGenreProfileResult,
        genre::get_genre_profile(genre::Genre::General)
    );
    emit!(
        DetectThemesSmartResult,
        theme::detect_themes_smart(&[], &[])
    );
    emit!(AnalyzeResolutionResult, theme::analyze_resolution(&[]));
    emit!(
        ExplainReadabilityResult,
        explain::explain_readability(0.0, 0.0, 1.0, "general")
    );
    emit!(
        ExplainPacingResult,
        explain::explain_pacing(0.0, 0, false, 0.0)
    );
    emit!(
        ExplainTensionResult,
        explain::explain_tension(0.0, 0.0, 0, 0, 0, 0, 0.0)
    );
    emit!(
        ExplainToneShiftResult,
        explain::explain_tone_shift("", "", 0, "general")
    );
    emit!(ExplainWhiteRoomResult, explain::explain_white_room(0, 0, 0));
    emit!(
        ExplainShowDontTellResult,
        explain::explain_show_dont_tell("", "")
    );
    emit!(
        ComputeSceneConfidenceResult,
        confidence::compute_scene_confidence("", 0)
    );
    emit!(
        ComputeReadabilityConfidenceResult,
        confidence::compute_readability_confidence(0, 0)
    );
    emit!(
        WordFrequenciesResult,
        word_stats::word_frequencies("", 0, |_| false)
    );
    emit!(
        WordFrequenciesFromWordsResult,
        word_stats::word_frequencies_from_words(vec![], 0, |_| false)
    );
    let mut types = String::from(
        "// Generated by cargo run --features bindings --example export_standalone_contracts.\n// Edit Rust wire types and regenerate.\n\n",
    );
    for declaration in contracts
        .definitions
        .into_values()
        .chain(contracts.aliases.into_values())
    {
        types.push_str(&declaration);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (path, contents) in [
        (root.join("standalone.generated.d.ts"), types),
        (
            root.join("docs/standalone.schema.json"),
            format!(
                "{}\n",
                serde_json::to_string_pretty(&contracts.schemas).unwrap()
            ),
        ),
    ] {
        if check {
            assert_eq!(
                fs::read_to_string(&path).expect("generated standalone contract missing"),
                contents,
                "{} is stale; run export_standalone_contracts",
                path.display()
            );
        } else {
            fs::write(path, contents).unwrap();
        }
    }
}
