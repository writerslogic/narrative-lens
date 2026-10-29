use narrative_lens::substrate::{confidence::*, utils::AnalysisAvailability};

#[test]
fn unicode_letters_do_not_lose_data_quality_to_utf8_byte_length() {
    let ascii = compute_scene_confidence("aaaa bbbb", 0);
    let unicode = compute_scene_confidence("éééé 中文中文", 0);
    assert_eq!(ascii.data_quality, unicode.data_quality);
    assert_eq!(ascii.overall, unicode.overall);
    for text in ["😀 😀", "… …", " "] {
        let measured = compute_scene_confidence(text, 0);
        assert_eq!(measured.data_quality, 0.0);
        assert_eq!(measured.overall, 0.0);
    }
}

#[test]
fn signal_subsets_and_finite_scene_indices_are_checked_at_boundaries() {
    for count in [3, 4, 5] {
        assert_eq!(
            compute_finding_confidence_checked(count, 4, 1.0).is_ok(),
            count <= 4
        );
    }
    for value in [
        -f64::EPSILON,
        0.0,
        f64::EPSILON,
        1.0 - f64::EPSILON,
        1.0,
        1.0 + f64::EPSILON,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let result = compute_finding_confidence_checked(1, 1, value);
        assert_eq!(
            result.is_ok(),
            value.is_finite() && (0.0..=1.0).contains(&value)
        );
        if let Ok(value) = result {
            assert!(value.is_finite() && (0.0..=1.0).contains(&value));
        }
    }
    assert!(compute_finding_confidence_checked(1, 0, 0.5).is_err());
    assert_eq!(compute_finding_confidence(1, 0, 0.5), 0.0);
}

#[test]
fn absent_signals_are_distinct_from_a_measured_zero_agreement_index() {
    let absent = measure_finding_confidence(0, 0, 0.0).unwrap();
    let measured = measure_finding_confidence(0, 1, 0.0).unwrap();
    assert_eq!(absent.value, None);
    assert_eq!(absent.availability, AnalysisAvailability::Unavailable);
    assert_eq!(measured.value, Some(0.0));
    assert_eq!(measured.availability, AnalysisAvailability::Complete);
    assert!(!measured.calibrated);
    assert_eq!(measured.confidence_basis, ConfidenceBasis::SignalAgreement);
}
