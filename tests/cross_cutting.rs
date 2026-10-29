use narrative_lens::{
    continuity::{causal_engine, coreference::CoreferenceResolver, epistemic_engine, mental_model},
    craft::{readability, subtext, syntax_tension, voice},
    structure::{
        genre_compliance, narrative_entropy, opening_analysis, promise_payoff, structure,
        thematic_argument,
    },
    substrate::{
        embedding::{CorpusEmbedding, EmbeddingOptions, SemanticIndex},
        text,
        utils::AnalysisAvailability,
    },
};

#[test]
fn original_sentence_and_paragraph_contracts_agree() {
    for sample in [
        "She waited. Then she left",
        "Dr. Smith waited. Then left",
        "Wait… really? Yes!",
        "She waited.\u{3000}Then she left",
        "\"Leave!\" She waited.",
    ] {
        let spans = narrative_lens::substrate::utils::sentence_spans(sample);
        assert_eq!(text::count_sentences(sample), spans.len());
        assert_eq!(spans.len(), if sample.starts_with("Wait") { 3 } else { 2 });
        for (start, end) in spans {
            assert!(!sample[start..end].trim().is_empty());
        }
    }
    for word in ["mistake", "explore", "secure", "marine", "sunrise"] {
        assert_eq!(text::count_syllables(word), 2, "{word}");
    }
    let lf = "😀 She waited in the garden.\n\nThen she went home.\n  \nRain fell.";
    let crlf = lf.replace('\n', "\r\n");
    for sample in [lf, &crlf] {
        let paragraphs = text::split_paragraphs(sample);
        let spans = text::paragraph_spans(sample);
        assert_eq!(paragraphs.len(), 3);
        assert_eq!(paragraphs.len(), spans.len());
        let encoded: Vec<_> = sample.encode_utf16().collect();
        for (paragraph, (quote, start, end)) in paragraphs.iter().zip(spans) {
            assert_eq!(*paragraph, quote);
            assert_eq!(
                String::from_utf16(&encoded[start as usize..end as usize]).unwrap(),
                quote
            );
        }
    }
    let lf_readability = readability::compute_readability(lf);
    let crlf_readability = readability::compute_readability(&crlf);
    assert_eq!(
        lf_readability.readability_variance,
        crlf_readability.readability_variance
    );
    assert_eq!(lf_readability.fkgl, crlf_readability.fkgl);
    assert_eq!(
        lf_readability.sentence_count,
        crlf_readability.sentence_count
    );
    assert_eq!(
        serde_json::to_value(syntax_tension::analyze_scene_text(lf)).unwrap(),
        serde_json::to_value(syntax_tension::analyze_scene_text(&crlf)).unwrap()
    );
}

#[test]
fn unicode_case_expansion_never_moves_coreference_mentions() {
    let sample = "İ Alice smiled. She waited.";
    let result = CoreferenceResolver::new().resolve(sample, &["Alice"], None);
    let alice = result
        .mentions
        .iter()
        .find(|mention| !mention.is_pronoun)
        .unwrap();
    assert_eq!(alice.start, 3);
    for mention in result.mentions {
        assert_eq!(
            &sample[mention.start..mention.start + mention.text.len()],
            mention.text
        );
    }
}

#[test]
fn voice_ties_and_sentence_lengths_are_reproducible() {
    let lines = vec!["cedar birch apple maple willow ash".into()];
    let expected = voice::build_voice_profile("A", &lines);
    assert_eq!(
        expected.top_words,
        ["apple", "ash", "birch", "cedar", "maple"]
    );
    for _ in 0..100 {
        assert_eq!(
            serde_json::to_value(voice::build_voice_profile("A", &lines)).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
    }
    let profile = voice::build_voice_profile("A", &["Zoë left. Zoë returned.".into()]);
    assert_eq!(profile.avg_sentence_length, 2.0);
    assert_eq!(profile.avg_word_length, 4.5);
}

#[test]
fn structure_and_causal_scores_share_their_declared_ranges() {
    let report =
        structure::validate_structure("nonlinear", vec![0.1, 0.5, 0.9, 0.4], vec![]).unwrap();
    assert_eq!(report.health_pct, 100.0);
    let scenes = [
        "Alice opened the gate. Alice crossed the bridge. Alice reached the tower. Alice lit the lamp.",
        "Alice decided to leave.",
    ];
    let report = causal_engine::analyze_causality(&scenes, &["Alice"]);
    assert_eq!(report.plot_coherence, 1.0);
    for _ in 0..50 {
        assert_eq!(
            serde_json::to_value(causal_engine::analyze_causality(&scenes, &["Alice"])).unwrap(),
            serde_json::to_value(&report).unwrap()
        );
    }
}

#[test]
fn unsupported_judgments_are_explicit_and_do_not_claim_success() {
    assert_eq!(
        subtext::analyze_subtext(&["I'm fine."], &[], &[], "literary").availability,
        AnalysisAvailability::Unavailable
    );
    assert_eq!(
        thematic_argument::track_thematic_argument(&["She left."], &[], &[], &[], "literary")
            .availability,
        AnalysisAvailability::Unavailable
    );
    assert_eq!(
        mental_model::build_mental_model(&["She left."], &[], &[], &[]).availability,
        AnalysisAvailability::Unavailable
    );
    let genre = genre_compliance::check_genre_compliance(
        "romance", true, false, false, true, true, false, 80_000, 1.0, "rising",
    );
    let connection = genre
        .conventions
        .iter()
        .find(|convention| convention.name == "Central Love Story")
        .unwrap();
    assert!(matches!(
        connection.status,
        genre_compliance::ConventionStatus::Unknown
    ));
    assert_eq!(genre.availability, AnalysisAvailability::Partial);
    let entropy =
        narrative_entropy::compute_narrative_entropy(&[], &[], "three_act", "literary", &[]);
    assert_eq!(entropy.availability, AnalysisAvailability::Unavailable);
    assert!(matches!(
        entropy.pacing_diagnosis,
        narrative_entropy::EntropyDiagnosis::Unavailable
    ));
    let opening = "She waited at the gate. The rain fell.";
    assert_eq!(
        opening_analysis::analyze_opening(opening, "literary", None).voice_establishment,
        opening_analysis::analyze_opening(opening, "literary", Some(&[1.0])).voice_establishment
    );
}

#[test]
fn promise_ledger_preserves_unknown_resolution_and_location() {
    let scenes = [
        "Who stole the silver key?",
        "Alice stole the silver key.",
        "Alice returned it.",
    ];
    let report = promise_payoff::build_promise_payoff_ledger(
        &scenes,
        &[5, 5, 3],
        "general",
        &[("silver key".into(), 0, true)],
        &[],
        "three_act",
        3,
    );
    assert_eq!(report.broken_count, 0);
    assert!(report.unearned_payoffs.is_empty());
    let question = report
        .promises
        .iter()
        .find(|promise| {
            matches!(
                promise.promise_type,
                promise_payoff::PromiseType::ExplicitQuestion(_)
            )
        })
        .unwrap();
    assert!(matches!(
        question.status,
        promise_payoff::PayoffStatus::Unknown
    ));
    let confirmed = report
        .promises
        .iter()
        .find(|promise| {
            matches!(
                promise.promise_type,
                promise_payoff::PromiseType::Foreshadowing(_)
            )
        })
        .unwrap();
    assert!(matches!(
        confirmed.status,
        promise_payoff::PayoffStatus::ConfirmedFulfilled
    ));
    assert!(confirmed.payoff_scene.is_none() && confirmed.payoff_quality.is_none());
    assert!(report.cascade_payoffs.is_empty());
}

#[test]
fn edited_rare_terms_update_lexical_retrieval_with_dense_scenes() {
    let documents = vec![
        ("a".into(), "The quorvex artifact was hidden.".into()),
        ("b".into(), "The garden bloomed.".into()),
    ];
    let options = EmbeddingOptions {
        min_count: 2,
        ..Default::default()
    };
    let mut index = SemanticIndex::build(&documents, &options);
    assert_eq!(index.search("quorvex", 5)[0].0, "a");
    index.update_document("a", "The zynthex artifact was hidden.");
    assert!(index.search("quorvex", 5).is_empty());
    assert_eq!(index.search("zynthex", 5)[0].0, "a");
    index.update_document("c", "Another zynthex artifact.");
    assert_eq!(index.search("zynthex", 5).len(), 2);
    for maximum in [4095, 4096] {
        let bounded = EmbeddingOptions {
            max_vocab: maximum,
            min_count: 1,
            ..Default::default()
        };
        assert!(CorpusEmbedding::train_checked(&["apple cedar birch"], &bounded).is_ok());
    }
    let mut invalid = options;
    invalid.max_vocab = 4097;
    assert!(CorpusEmbedding::train_checked(&["one two three"], &invalid).is_err());
    assert_eq!(
        CorpusEmbedding::train(&["one two three"], &invalid).dim(),
        0
    );
}

#[test]
fn repeated_explicit_facts_do_not_create_new_revelations() {
    let sample = "Alice discovered a secret passage.";
    let result =
        epistemic_engine::track_epistemics(&[sample, sample], &["Alice"], &["Alice", "Alice"]);
    assert_eq!(result.facts.len(), 1);
    assert!(
        result
            .revelations
            .iter()
            .all(|revelation| revelation.scene == 0)
    );
    assert_eq!(result.availability, AnalysisAvailability::Partial);
}
