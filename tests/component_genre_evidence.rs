use narrative_lens::structure::genre_compliance::{ConventionStatus, check_genre_compliance};

#[test]
fn structural_health_does_not_prove_prose_quality_or_interiority() {
    let result = check_genre_compliance(
        "literary", false, false, false, true, true, false, 80000, 0.99, "rising",
    );
    assert!(
        result
            .conventions
            .iter()
            .filter(|c| c.name == "Prose Quality" || c.name == "Character Interiority")
            .all(|c| matches!(c.status, ConventionStatus::Unknown))
    );
}

#[test]
fn survival_false_does_not_prove_intentional_subversion() {
    let result = check_genre_compliance(
        "thriller", false, false, true, false, true, false, 80000, 0.99, "rising",
    );
    let survival = result
        .conventions
        .iter()
        .find(|c| c.name == "Protagonist Survives")
        .unwrap();
    assert!(matches!(survival.status, ConventionStatus::NotMet));
}

#[test]
fn coarse_inputs_do_not_assert_a_secondary_genre() {
    let result = check_genre_compliance(
        "mystery", true, true, true, true, true, false, 80000, 0.99, "rising",
    );
    assert!(!result.is_genre_blend);
    assert!(result.secondary_genres.is_empty());
}

use narrative_lens::structure::genre_compliance::{GenreEvidence, check_genre_compliance_checked};
use narrative_lens::substrate::utils::AnalysisAvailability;

#[test]
fn absent_evidence_stays_unknown_for_every_supported_checklist() {
    let empty: GenreEvidence = serde_json::from_str("{}").unwrap();
    for genre in [
        "romance",
        "mystery",
        "thriller",
        "literary",
        "fantasy",
        "sci-fi",
        "horror",
        "historical",
        "ya",
        "women's fiction",
        "upmarket",
        "middle-grade",
    ] {
        let result = check_genre_compliance_checked(genre, &empty).unwrap();
        assert!(!result.conventions.is_empty());
        assert_eq!(result.availability, AnalysisAvailability::Unavailable);
        assert_eq!(result.unknown_count, result.conventions.len());
        assert_eq!(result.assessed_count, 0);
        assert_eq!(result.evidence_coverage, 0.0);
        assert!(
            result
                .conventions
                .iter()
                .all(|c| c.status == ConventionStatus::Unknown)
        );
        assert!(!result.quality_assessed);
    }
    assert!(serde_json::from_str::<GenreEvidence>(r#"{"unexpected":true}"#).is_err());
}

#[test]
fn direct_false_is_distinct_from_absent_and_counts_expose_coverage() {
    for value in [None, Some(false), Some(true)] {
        let result = check_genre_compliance_checked(
            "romance",
            &GenreEvidence {
                has_hea: value,
                ..Default::default()
            },
        )
        .unwrap();
        let status = &result
            .conventions
            .iter()
            .find(|c| c.name == "HEA/HFN")
            .unwrap()
            .status;
        assert_eq!(
            *status,
            match value {
                None => ConventionStatus::Unknown,
                Some(false) => ConventionStatus::NotMet,
                Some(true) => ConventionStatus::Met,
            }
        );
        assert_eq!(result.assessed_count, usize::from(value.is_some()));
        assert_eq!(
            result.unknown_count + result.assessed_count,
            result.conventions.len()
        );
        if value.is_some() {
            assert_eq!(result.availability, AnalysisAvailability::Partial);
            assert!(
                (result.evidence_coverage - 1.0 / result.conventions.len() as f64).abs() < 1e-12
            );
        }
    }
}

#[test]
fn explicit_labels_support_partial_and_subverted_without_inferred_intent() {
    let empty = check_genre_compliance_checked("thriller", &GenreEvidence::default()).unwrap();
    let mut evidence = GenreEvidence::default();
    for c in empty.conventions {
        evidence
            .convention_judgments
            .insert(c.name, ConventionStatus::PartiallyMet);
    }
    evidence
        .convention_judgments
        .insert("Protagonist Survives".into(), ConventionStatus::Subverted);
    let result = check_genre_compliance_checked("thriller", &evidence).unwrap();
    assert_eq!(result.availability, AnalysisAvailability::Complete);
    assert_eq!(result.evidence_coverage, 1.0);
    assert!(!result.quality_assessed);
    assert_eq!(result.judgment_provenance, "caller_supplied");
    assert_eq!(result.present_count, 1);
    assert!(
        result
            .conventions
            .iter()
            .all(|c| c.evidence.contains("Caller-supplied"))
    );
}

#[test]
fn contradictory_unknown_or_irrelevant_judgments_are_rejected_secret_free() {
    let mut evidence = GenreEvidence {
        has_hea: Some(false),
        ..Default::default()
    };
    evidence
        .convention_judgments
        .insert("HEA/HFN".into(), ConventionStatus::Met);
    assert!(check_genre_compliance_checked("romance", &evidence).is_err());
    evidence
        .convention_judgments
        .insert("HEA/HFN".into(), ConventionStatus::NotMet);
    assert!(check_genre_compliance_checked("romance", &evidence).is_ok());
    assert!(check_genre_compliance_checked("mystery", &evidence).is_err());
    evidence.convention_judgments.clear();
    evidence.convention_judgments.insert(
        "private-unrecognized-input".repeat(1000),
        ConventionStatus::Met,
    );
    let error = check_genre_compliance_checked("romance", &evidence)
        .unwrap_err()
        .to_string();
    assert!(!error.contains("private-unrecognized-input"));
    for genre in ["", "unknown", "romantic mystery", "private-input"] {
        let error = check_genre_compliance_checked(genre, &GenreEvidence::default())
            .unwrap_err()
            .to_string();
        assert!(!error.contains("private-input"));
    }
}

#[test]
fn aliases_preserve_checklist_and_direct_evidence() {
    for (canonical, alias) in [
        ("sci-fi", "Science Fiction"),
        ("literary", " Literary Fiction "),
        ("historical", "historical fiction"),
        ("ya", "young adult"),
        ("middle-grade", "middle grade"),
    ] {
        let evidence = GenreEvidence {
            worldbuilding_present: Some(true),
            character_growth: Some(false),
            ..Default::default()
        };
        assert_eq!(
            serde_json::to_value(check_genre_compliance_checked(canonical, &evidence).unwrap())
                .unwrap(),
            serde_json::to_value(check_genre_compliance_checked(alias, &evidence).unwrap())
                .unwrap()
        );
    }
}

#[test]
fn structural_signals_are_preserved_without_semantic_promotions_and_validate_units() {
    for score in [
        -f64::EPSILON,
        0.0,
        1.0,
        1.0 + f64::EPSILON,
        f64::NAN,
        f64::INFINITY,
    ] {
        let evidence = GenreEvidence {
            structure_health: Some(score),
            ..Default::default()
        };
        assert_eq!(
            check_genre_compliance_checked("literary", &evidence).is_ok(),
            score.is_finite() && (0.0..=1.0).contains(&score)
        );
    }
    for words in [64 * 1024 * 1024 - 1, 64 * 1024 * 1024, 64 * 1024 * 1024 + 1] {
        assert_eq!(
            check_genre_compliance_checked(
                "ya",
                &GenreEvidence {
                    word_count: Some(words),
                    ..Default::default()
                }
            )
            .is_ok(),
            words <= 64 * 1024 * 1024
        );
    }
    let result = check_genre_compliance_checked(
        "horror",
        &GenreEvidence {
            word_count: Some(80000),
            structure_health: Some(0.9),
            tension_arc_shape: Some("rising".into()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.structural_signals.structure_health, Some(0.9));
    assert_eq!(result.assessed_count, 0);
    assert_eq!(result.unknown_count, result.conventions.len());
    assert!(
        check_genre_compliance_checked(
            "horror",
            &GenreEvidence {
                tension_arc_shape: Some("unrecognized".into()),
                ..Default::default()
            }
        )
        .is_err()
    );
}
