use narrative_lens::{
    continuity::{gap_analysis as gaps, timeline},
    substrate::utils::AnalysisAvailability,
};

#[test]
fn gap_locations_are_bounded_and_reports_remain_review_candidates() {
    let result = gaps::analyze_gaps_checked(
        &[("late promise".into(), 2)],
        &[("Alice".into(), 2)],
        &[(0, 2)],
        &[("intentional ambiguity".into(), 2)],
        3,
        1,
    )
    .unwrap();
    assert_eq!(result.total_gap_count, 4);
    assert_eq!(result.availability, AnalysisAvailability::Partial);
    for gap in &result.gaps {
        assert!(gap.description.starts_with("Supplied"));
        assert!(gap.location.after_scene < 3);
        if let Some(before) = gap.location.before_scene {
            assert!(before > gap.location.after_scene && before < 3);
        }
    }
    let empty = gaps::analyze_gaps_checked(&[], &[], &[], &[], 0, 0).unwrap();
    assert!(empty.gaps.is_empty());
    for position in [3, usize::MAX] {
        assert!(
            gaps::analyze_gaps_checked(&[("x".into(), position)], &[], &[], &[], 3, 1).is_err()
        );
        assert!(
            gaps::analyze_gaps_checked(&[], &[("Alice".into(), position)], &[], &[], 3, 1).is_err()
        );
        assert!(
            gaps::analyze_gaps_checked(&[], &[], &[], &[("x".into(), position)], 3, 1).is_err()
        );
        assert!(gaps::analyze_gaps_checked(&[], &[], &[], &[], 3, position).is_err());
    }
    for range in [(2, 1), (1, 1), (0, 3), (0, usize::MAX)] {
        assert!(gaps::analyze_gaps_checked(&[], &[], &[range], &[], 3, 1).is_err());
    }
    assert!(gaps::analyze_gaps_checked(&[("x".into(), 0)], &[], &[], &[], 0, 0).is_err());
    let unavailable = gaps::analyze_gaps(&[("x".into(), usize::MAX)], &[], &[], &[], 1, 0);
    assert_eq!(unavailable.availability, AnalysisAvailability::Unavailable);
    assert!(!unavailable.diagnostics.is_empty());
}

#[test]
fn gap_projection_is_permutation_stable_and_enforces_exact_limits() {
    let a = [("z".into(), 0), ("a".into(), 0), ("b".into(), 1)];
    let b = [a[2].clone(), a[1].clone(), a[0].clone()];
    assert_eq!(
        serde_json::to_value(gaps::analyze_gaps_checked(&a, &[], &[], &[], 3, 2).unwrap()).unwrap(),
        serde_json::to_value(gaps::analyze_gaps_checked(&b, &[], &[], &[], 3, 2).unwrap()).unwrap()
    );
    for count in [
        gaps::MAX_GAP_SIGNALS - 1,
        gaps::MAX_GAP_SIGNALS,
        gaps::MAX_GAP_SIGNALS + 1,
    ] {
        let signals = vec![("x".into(), 0); count];
        assert_eq!(
            gaps::analyze_gaps_checked(&signals, &[], &[], &[], 1, 0).is_ok(),
            count <= gaps::MAX_GAP_SIGNALS
        );
    }
    for count in [
        gaps::MAX_GAP_SCENES - 1,
        gaps::MAX_GAP_SCENES,
        gaps::MAX_GAP_SCENES + 1,
    ] {
        assert_eq!(
            gaps::analyze_gaps_checked(&[], &[], &[], &[], count, 0).is_ok(),
            count <= gaps::MAX_GAP_SCENES
        );
    }
    for bytes in [4095, 4096, 4097] {
        assert_eq!(
            gaps::analyze_gaps_checked(&[("x".repeat(bytes), 0)], &[], &[], &[], 1, 0).is_ok(),
            bytes <= 4096
        );
    }
    for bytes in [
        gaps::MAX_GAP_METADATA_BYTES - 1,
        gaps::MAX_GAP_METADATA_BYTES,
        gaps::MAX_GAP_METADATA_BYTES + 1,
    ] {
        let mut signals = vec![("x".repeat(4096), 0); bytes / 4096];
        if bytes % 4096 != 0 {
            signals.push(("x".repeat(bytes % 4096), 0));
        }
        assert_eq!(
            gaps::analyze_gaps_checked(&signals, &[], &[], &[], 1, 0).is_ok(),
            bytes <= gaps::MAX_GAP_METADATA_BYTES
        );
    }
}

#[test]
fn timeline_calendar_checks_skip_invalid_candidates_without_reordering_scenes() {
    let samples = [
        ("99:99 then 23:59", Some("23:59")),
        ("24:00 then 00:00", Some("00:00")),
        ("February 30, 2020 then Monday", Some("Monday")),
        ("February 29, 1900 then Tuesday", Some("Tuesday")),
        ("February 29, 2000", Some("February 29, 2000")),
        ("April 31 then May 5", Some("May 5")),
        ("March 0 then 9:30", Some("9:30")),
        ("You may go now", None),
        ("\u{feff}March\n1847", Some("March 1847")),
    ];
    let documents: Vec<_> = samples
        .iter()
        .enumerate()
        .map(|(i, (text, _))| (format!("d{i}"), (*text).into()))
        .collect();
    let entries = timeline::build_timeline_checked(&documents).unwrap();
    for (i, (entry, (_, expected))) in entries.iter().zip(samples).enumerate() {
        assert_eq!(entry.order, i);
        assert_eq!(entry.document_id, format!("d{i}"));
        assert_eq!(entry.time_hint.as_deref(), expected);
    }
    let backward = timeline::build_timeline_checked(&[
        ("later".into(), "1914".into()),
        ("earlier".into(), "1847".into()),
    ])
    .unwrap();
    assert_eq!(backward[0].document_id, "later");
    assert_eq!(backward[1].document_id, "earlier");
}

#[test]
fn timeline_patches_are_last_field_wins_idempotent_and_atomic() {
    let mut entries = timeline::build_timeline_checked(&[("d".into(), "Monday".into())]).unwrap();
    let patches = [
        timeline::TimelineOverride {
            document_id: "d".into(),
            lane: Some("Alice".into()),
            time_hint: Some("yesterday".into()),
            ..Default::default()
        },
        timeline::TimelineOverride {
            document_id: "d".into(),
            label: Some("final".into()),
            time_hint: Some(" ".into()),
            ..Default::default()
        },
        timeline::TimelineOverride {
            document_id: "d".into(),
            lane: Some("Bob".into()),
            ..Default::default()
        },
    ];
    timeline::apply_overrides_checked(&mut entries, &patches).unwrap();
    assert_eq!(entries[0].lane.as_deref(), Some("Bob"));
    assert_eq!(entries[0].label.as_deref(), Some("final"));
    assert_eq!(entries[0].time_hint.as_deref(), Some("yesterday"));
    let expected = entries.clone();
    timeline::apply_overrides_checked(&mut entries, &patches).unwrap();
    assert_eq!(entries, expected);
    let invalid = [
        patches[0].clone(),
        timeline::TimelineOverride {
            document_id: "missing".into(),
            ..Default::default()
        },
    ];
    assert!(timeline::apply_overrides_checked(&mut entries, &invalid).is_err());
    assert_eq!(entries, expected);
    entries[0].order = usize::MAX;
    assert!(timeline::apply_overrides_checked(&mut entries, &patches).is_err());
    assert_eq!(entries[0].lane.as_deref(), Some("Bob"));
}

#[test]
fn timeline_checks_document_and_metadata_boundaries_before_mutation() {
    assert!(
        timeline::build_timeline_checked(&[("d".into(), "".into()), ("d".into(), "".into())])
            .is_err()
    );
    for count in [9999, 10_000, 10_001] {
        let docs: Vec<_> = (0..count)
            .map(|i| (format!("d{i}"), String::new()))
            .collect();
        assert_eq!(
            timeline::build_timeline_checked(&docs).is_ok(),
            count <= timeline::MAX_TIMELINE_DOCUMENTS
        );
        let patches = vec![
            timeline::TimelineOverride {
                document_id: "d".into(),
                ..Default::default()
            };
            count
        ];
        let mut entries = timeline::build_timeline(&[("d".into(), "".into())]);
        assert_eq!(
            timeline::apply_overrides_checked(&mut entries, &patches).is_ok(),
            count <= timeline::MAX_TIMELINE_OVERRIDES
        );
    }
    for bytes in [255, 256, 257] {
        assert_eq!(
            timeline::build_timeline_checked(&[("d".repeat(bytes), "".into())]).is_ok(),
            bytes <= 256
        );
    }
    for bytes in [4095, 4096, 4097] {
        let mut entries = timeline::build_timeline(&[("d".into(), "".into())]);
        let patch = timeline::TimelineOverride {
            document_id: "d".into(),
            notes: Some("x".repeat(bytes)),
            ..Default::default()
        };
        assert_eq!(
            timeline::apply_overrides_checked(&mut entries, &[patch]).is_ok(),
            bytes <= timeline::MAX_TIMELINE_FIELD_BYTES
        );
    }
    for bytes in [
        timeline::MAX_TIMELINE_METADATA_BYTES - 1,
        timeline::MAX_TIMELINE_METADATA_BYTES,
        timeline::MAX_TIMELINE_METADATA_BYTES + 1,
    ] {
        let mut entries = timeline::build_timeline(&[("d".into(), "".into())]);
        // Entry id plus one patch id per 4096-byte note are included in the budget.
        let mut remaining = bytes - 1;
        let mut patches = Vec::new();
        while remaining > 0 {
            let note_bytes = remaining.saturating_sub(1).min(4096);
            patches.push(timeline::TimelineOverride {
                document_id: "d".into(),
                notes: Some("x".repeat(note_bytes)),
                ..Default::default()
            });
            remaining -= note_bytes + 1;
        }
        assert_eq!(
            timeline::apply_overrides_checked(&mut entries, &patches).is_ok(),
            bytes <= timeline::MAX_TIMELINE_METADATA_BYTES
        );
    }
}

#[test]
fn timeline_validates_text_byte_budgets_at_exact_boundaries() {
    for bytes in [
        timeline::MAX_TIMELINE_TEXT_BYTES - 1,
        timeline::MAX_TIMELINE_TEXT_BYTES,
        timeline::MAX_TIMELINE_TEXT_BYTES + 1,
    ] {
        let documents = [("d".into(), "x".repeat(bytes))];
        assert_eq!(
            timeline::build_timeline_checked(&documents).is_ok(),
            bytes <= timeline::MAX_TIMELINE_TEXT_BYTES
        );
    }
    for bytes in [
        timeline::MAX_TIMELINE_TOTAL_BYTES - 1,
        timeline::MAX_TIMELINE_TOTAL_BYTES,
        timeline::MAX_TIMELINE_TOTAL_BYTES + 1,
    ] {
        let mut documents = Vec::new();
        let mut remaining = bytes;
        while remaining > 0 {
            let length = remaining.min(timeline::MAX_TIMELINE_TEXT_BYTES);
            documents.push((format!("d{}", documents.len()), "x".repeat(length)));
            remaining -= length;
        }
        assert_eq!(
            timeline::build_timeline_checked(&documents).is_ok(),
            bytes <= timeline::MAX_TIMELINE_TOTAL_BYTES
        );
    }
}
