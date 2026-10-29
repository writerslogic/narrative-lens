use narrative_lens::continuity::world_state::*;

#[test]
fn named_lock_uses_the_actors_previously_observed_location() {
    let result = track_world_state(
        &[
            "Alice was in the cellar.",
            "Alice locked the door.",
            "Alice entered the garden.",
        ],
        &["Alice"],
    );
    assert_eq!(result.violations.len(), 1, "{:#?}", result.violations);
    assert_eq!(
        result.violations[0].violation_type,
        ViolationType::ImpossibleLocation
    );
}

#[test]
fn named_unlock_cannot_release_a_constraint_in_someone_elses_location() {
    for bob_location in [None, Some("garden")] {
        let initial = if let Some(location) = bob_location {
            format!("Alice was in the cellar. Someone locked the door. Bob was in the {location}.")
        } else {
            "Alice was in the cellar. Someone locked the door.".into()
        };
        let result = track_world_state(
            &[
                &initial,
                "Bob unlocked the door.",
                "Alice entered the garden.",
            ],
            &["Bob", "Alice"],
        );
        assert_eq!(
            result.violations.len(),
            1,
            "Bob location {bob_location:?}: {:#?}",
            result.violations
        );
    }
}

fn assert_equivalent(scenes: &[&str], cast: &[&str]) {
    let full = track_world_state(scenes, cast);
    let final_facts = full
        .snapshots
        .last()
        .map(|snapshot| snapshot.facts.clone())
        .unwrap_or_default();
    for include_deltas in [false, true] {
        for include_event_history in [false, true] {
            let options = WorldStateOptions {
                include_deltas,
                include_event_history,
            };
            let current = track_world_state_current(scenes, cast, &options).unwrap();
            assert_eq!(current.options, options);
            assert_eq!(current.availability, full.availability);
            assert_eq!(current.diagnostics, full.diagnostics);
            assert_eq!(current.facts, final_facts);
            assert_eq!(
                current.after_scene,
                full.snapshots.last().map(|snapshot| snapshot.after_scene)
            );
            assert_eq!(current.continuity_score, full.continuity_score);
            assert_eq!(current.world_complexity, full.world_complexity);
            assert_eq!(
                serde_json::to_value(&current.violations).unwrap(),
                serde_json::to_value(&full.violations).unwrap()
            );
            let mut expected_knowledge = full.character_knowledge.clone();
            if !include_event_history {
                for character in &mut expected_knowledge {
                    character.locations_visited.clear();
                    character.told_by.clear();
                }
            }
            assert_eq!(
                serde_json::to_value(&current.character_knowledge).unwrap(),
                serde_json::to_value(expected_knowledge).unwrap()
            );
            if include_event_history {
                assert_eq!(current.location_timeline, full.location_timeline);
                assert_eq!(current.information_flow, full.information_flow);
            } else {
                assert!(current.location_timeline.is_empty());
                assert!(current.information_flow.is_empty());
            }
            if include_deltas {
                let mut replay = std::collections::HashSet::new();
                assert_eq!(current.deltas.len(), full.snapshots.len());
                for (delta, snapshot) in current.deltas.iter().zip(&full.snapshots) {
                    assert_eq!(delta.after_scene, snapshot.after_scene);
                    let mut added: Vec<_> = snapshot.facts.difference(&replay).cloned().collect();
                    let mut removed: Vec<_> = replay.difference(&snapshot.facts).cloned().collect();
                    added.sort();
                    removed.sort();
                    assert_eq!(delta.new_facts, added);
                    assert_eq!(delta.invalidated_facts, removed);
                    for fact in &delta.invalidated_facts {
                        assert!(replay.remove(fact));
                    }
                    for fact in &delta.new_facts {
                        assert!(replay.insert(fact.clone()));
                    }
                    assert_eq!(replay, snapshot.facts, "scene {}", delta.after_scene);
                }
                assert_eq!(replay, current.facts);
            } else {
                assert!(current.deltas.is_empty());
            }
        }
    }
}

#[test]
fn current_and_net_deltas_equal_full_reduction_for_empty_and_unknown_inputs() {
    assert_equivalent(&[], &[]);
    assert_equivalent(&[], &["Alice"]);
    assert_equivalent(&["Rain fell."], &[]);
    assert_equivalent(
        &["Rain fell.", "Clara learned that a comet arrived.", ""],
        &["Alice"],
    );
    let current = track_world_state_current(
        &["Rain fell.", ""],
        &["Alice"],
        &WorldStateOptions {
            include_deltas: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(current.after_scene, Some(1));
    assert_eq!(current.deltas.len(), 2);
    assert!(
        current
            .deltas
            .iter()
            .all(|delta| delta.new_facts.is_empty() && delta.invalidated_facts.is_empty())
    );
    assert!(current.character_knowledge[0].knows.is_empty());
    assert_eq!(current.continuity_score, 1.0);
}

#[test]
fn unicode_cast_reordering_and_duplicates_preserve_current_and_history() {
    let scenes = [
        "Éva was in the café. 李明 was in the café.",
        "Éva learned that 雪 is falling. Éva told 李明 that 雪 is falling.",
        "李明 entered the garden.",
    ];
    let options = WorldStateOptions {
        include_deltas: true,
        include_event_history: true,
    };
    let first = track_world_state_current(&scenes, &["Éva", "李明", "éva"], &options).unwrap();
    let second = track_world_state_current(&scenes, &["éva", "李明", "Éva"], &options).unwrap();
    assert_eq!(
        serde_json::to_value(first).unwrap(),
        serde_json::to_value(second).unwrap()
    );
    assert_equivalent(&scenes, &["Éva", "李明", "éva"]);
}

#[test]
fn net_deltas_cancel_within_scene_transfers_movements_and_lock_transitions() {
    let scenes = [
        "Alice carried the key. Bob was in the hall. Alice was in the hall. Alice locked the door. Alice unlocked the door.",
        "Alice handed the key to Bob. Bob gave the key to Alice. Alice entered the yard. Alice entered the hall. Alice locked the door. Alice unlocked the door.",
        "Alice handed the key to Bob. Alice locked the door.",
        "Alice unlocked the door. Bob entered the yard. Bob carried the key.",
    ];
    assert_equivalent(&scenes, &["Bob", "Alice"]);
    let current = track_world_state_current(
        &scenes,
        &["Alice", "Bob"],
        &WorldStateOptions {
            include_deltas: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(current.deltas[1].new_facts.is_empty());
    assert!(current.deltas[1].invalidated_facts.is_empty());
    assert!(!current.deltas[2].new_facts.is_empty());
    assert!(!current.deltas[2].invalidated_facts.is_empty());
    let full = track_world_state(&scenes, &["Alice", "Bob"]);
    assert!(!full.snapshots[1].new_facts.is_empty());
    assert!(!full.snapshots[1].invalidated_facts.is_empty());
    assert!(current.facts.contains(&WorldFact::CharacterPossesses {
        character: "bob".into(),
        object: "key".into()
    }));
    assert!(
        !current
            .facts
            .iter()
            .any(|fact| matches!(fact, WorldFact::PhysicalConstraint { .. }))
    );
}

#[test]
fn optional_event_history_does_not_change_learning_constraints_or_violations() {
    let scenes = [
        "Alice was in the cellar. Alice learned that the safe was empty. Alice locked the door.",
        "Alice told Bob that the safe was empty. Bob saw the bridge collapse.",
        "Alice entered the garden.",
        "Alice was freed. Alice entered the roof.",
    ];
    assert_equivalent(&scenes, &["Alice", "Bob"]);
    let current =
        track_world_state_current(&scenes, &["Alice", "Bob"], &WorldStateOptions::default())
            .unwrap();
    assert_eq!(current.violations.len(), 1);
    let bob = current
        .character_knowledge
        .iter()
        .find(|character| character.character == "Bob")
        .unwrap();
    assert!(!bob.knows.is_empty());
    assert!(!bob.witnessed.is_empty());
    assert!(bob.told_by.is_empty());
}

#[test]
fn options_default_reject_unknown_fields_and_round_trip_deterministically() {
    assert_eq!(
        serde_json::from_str::<WorldStateOptions>("{}").unwrap(),
        WorldStateOptions::default()
    );
    assert_eq!(
        serde_json::from_str::<WorldStateOptions>(r#"{"includeDeltas":true}"#).unwrap(),
        WorldStateOptions {
            include_deltas: true,
            ..Default::default()
        }
    );
    assert!(serde_json::from_str::<WorldStateOptions>(r#"{"unknown":true}"#).is_err());
    assert!(serde_json::from_str::<WorldStateOptions>(r#"{"includeDeltas":"true"}"#).is_err());
    let first = track_world_state_current(
        &["Alice carried the key. Alice learned that rain fell."],
        &["Alice"],
        &WorldStateOptions {
            include_deltas: true,
            ..Default::default()
        },
    )
    .unwrap();
    let encoded = serde_json::to_vec(&first).unwrap();
    let second: WorldStateCurrentResult = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(encoded, serde_json::to_vec(&second).unwrap());
}

#[test]
fn growing_fact_corpus_has_identical_scene_boundaries_and_final_state() {
    let scenes: Vec<_> = (0..128).map(|scene| format!("Alice learned that observation {scene} holds. Bob learned that report {scene} holds.")).collect();
    let refs: Vec<_> = scenes.iter().map(String::as_str).collect();
    assert_equivalent(&refs, &["Alice", "Bob"]);
}

#[test]
fn unconfigured_named_unlock_does_not_borrow_the_sole_lock_location() {
    for subject in ["Charlie", "Charlie Brown", "He", "The stranger"] {
        let unlock = format!("{subject} unlocked the door.");
        let scenes = [
            "Alice was in the cellar. Alice locked the door.",
            unlock.as_str(),
            "Alice entered the garden.",
        ];
        let current = track_world_state_current(
            &scenes,
            &["Alice", "Bob"],
            &WorldStateOptions {
                include_deltas: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(current.violations.len(), 1, "subject {subject}");
        assert!(current.deltas[1].invalidated_facts.is_empty());
        assert_equivalent(&scenes, &["Alice", "Bob"]);
    }
}

#[test]
fn unconfigured_named_lock_does_not_borrow_other_participants_location() {
    let scenes = [
        "Alice was in the cellar. Charlie locked the door.",
        "Alice entered the garden.",
    ];
    let current =
        track_world_state_current(&scenes, &["Alice", "Bob"], &WorldStateOptions::default())
            .unwrap();
    assert!(current.violations.is_empty());
    assert!(
        !current
            .facts
            .iter()
            .any(|fact| matches!(fact, WorldFact::PhysicalConstraint { .. }))
    );
}

#[test]
fn genuinely_unqualified_unlock_retains_legacy_single_location_fallback() {
    for unlock in [
        "Unlocked the door.",
        "Someone unlocked the door.",
        "Somebody unlocked the door.",
        "Night fell; unlocked the door.",
    ] {
        let scenes = [
            "Alice was in the cellar. Alice locked the door.",
            unlock,
            "Alice entered the garden.",
        ];
        assert!(
            track_world_state(&scenes, &["Alice"]).violations.is_empty(),
            "{unlock}"
        );
        assert_equivalent(&scenes, &["Alice"]);
    }
}

fn assert_validated(scenes: &[&str], cast: &[&str], valid: bool) {
    assert_eq!(track_world_state_checked(scenes, cast).is_ok(), valid);
    assert_eq!(
        track_world_state_current(scenes, cast, &WorldStateOptions::default()).is_ok(),
        valid
    );
}

#[test]
fn names_and_input_counts_enforce_existing_manuscript_limits() {
    for bytes in [0, 1, 2, 127, 128, 129] {
        let name = "x".repeat(bytes);
        assert_validated(&["A waited."], &[&name], (1..=128).contains(&bytes));
    }
    for name in ["  ", "Alice\n", "Alice\0", "Alice\u{0085}"] {
        assert_validated(&["A waited."], &[name], false);
    }
    let unicode = "é".repeat(64);
    assert_validated(&["A waited."], &[&unicode], true);
    assert_validated(&["A waited."], &[&format!("{unicode}x")], false);
    for count in [255, 256, 257] {
        let names: Vec<_> = (0..count).map(|i| format!("Actor{i}")).collect();
        let refs: Vec<_> = names.iter().map(String::as_str).collect();
        assert_validated(&["A waited."], &refs, count <= 256);
    }
    for count in [9_999, 10_000, 10_001] {
        let scenes = vec![""; count];
        assert_validated(&scenes, &["Alice"], count <= 10_000);
    }
}

#[test]
fn scene_and_total_byte_limits_cover_minus_equal_plus_one_without_copying_input() {
    let limit = 16 * 1024 * 1024;
    let source = " ".repeat(limit + 1);
    for bytes in [limit - 1, limit, limit + 1] {
        assert_validated(&[&source[..bytes]], &[], bytes <= limit);
    }
    let chunk = &source[..limit];
    assert_validated(&[chunk, chunk, chunk, &source[..limit - 1]], &[], true);
    assert_validated(&[chunk, chunk, chunk, chunk], &[], true);
    assert_validated(&[chunk, chunk, chunk, chunk, " "], &[], false);
}

#[test]
fn oversized_names_fail_without_abort_and_legacy_abstains_explicitly() {
    use narrative_lens::substrate::utils::AnalysisAvailability;
    let mut secret = "private-sensitive-name".repeat(10_000);
    secret.truncate(200_000);
    let error = track_world_state_checked(&["A waited."], &[&secret]).unwrap_err();
    assert_eq!(
        error.to_string(),
        "invalid input: character names must contain 1..=128 bytes and no control characters"
    );
    let legacy = track_world_state(&["A waited."], &[&secret]);
    assert_eq!(legacy.availability, AnalysisAvailability::Unavailable);
    assert_eq!(legacy.diagnostics, ["world_state_invalid_input"]);
    assert!(legacy.snapshots.is_empty());
    assert_eq!(legacy.continuity_score, 0.0);
    assert!(
        !serde_json::to_string(&legacy)
            .unwrap()
            .contains("sensitive")
    );
    let mut old = serde_json::to_value(track_world_state(&[], &[])).unwrap();
    old.as_object_mut().unwrap().remove("availability");
    old.as_object_mut().unwrap().remove("diagnostics");
    let imported: WorldStateResult = serde_json::from_value(old).unwrap();
    assert_eq!(imported.availability, AnalysisAvailability::Unavailable);
}
