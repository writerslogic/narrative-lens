use narrative_lens::continuity::entity::{
    self, CorrectionAction, EntityCorrection, EntityType, Mention,
};
use std::collections::BTreeSet;

#[test]
fn title_normalization_preserves_surnames_place_names_and_symbol_operators() {
    let distinct = entity::resolve_checked(
        &[
            Mention::new("John John", vec![]),
            Mention::new("John Smith", vec![]),
            Mention::new("Smith John", vec![]),
        ],
        EntityType::Character,
    )
    .unwrap();
    assert_eq!(distinct.len(), 3);
    assert_eq!(
        distinct
            .iter()
            .map(|entity| &entity.id)
            .collect::<BTreeSet<_>>()
            .len(),
        3
    );
    let mentions = [
        Mention::new("Stephen King", vec![]),
        Mention::new("Stephen Brown", vec![]),
        Mention::new("King", vec![]),
        Mention::new("Mr. King", vec![]),
    ];
    let entities = entity::resolve_checked(&mentions, EntityType::Character).unwrap();
    assert_eq!(entities.len(), 2);
    let king = entities
        .iter()
        .find(|entity| entity.canonical != "Stephen Brown")
        .unwrap();
    assert_eq!(king.mention_count, 3);
    for kind in [EntityType::Setting, EntityType::Other] {
        let entities = entity::resolve_checked(
            &[
                Mention::new("King's Landing", vec![]),
                Mention::new("Stephen Landing", vec![]),
            ],
            kind,
        )
        .unwrap();
        assert_eq!(entities.len(), 2);
        let entities = entity::resolve_checked(
            &[Mention::new("Bob", vec![]), Mention::new("Robert", vec![])],
            kind,
        )
        .unwrap();
        assert_eq!(entities.len(), 2);
    }
    let symbols = entity::resolve_checked(
        &[
            Mention::new("C++", vec![]),
            Mention::new("C#", vec![]),
            Mention::new("the C++", vec![]),
        ],
        EntityType::Symbol,
    )
    .unwrap();
    assert_eq!(symbols.len(), 2);
    assert!(symbols.iter().any(|entity| entity.mention_count == 2));
}

#[test]
fn zero_and_extreme_contexts_do_not_create_fake_negative_identity_evidence() {
    let zero = entity::resolve_checked(
        &[
            Mention::new("Alice Smith", vec![0.0, 0.0]),
            Mention::new("Alice Smith", vec![0.0, 0.0]),
        ],
        EntityType::Character,
    )
    .unwrap();
    assert_eq!(zero.len(), 1);
    assert_eq!(zero[0].mention_count, 2);
    let simple = entity::resolve_checked(
        &[
            Mention::new("Alice Smith", vec![1.0, 1.0]),
            Mention::new("Alice", vec![1.0, 1.0]),
        ],
        EntityType::Character,
    )
    .unwrap();
    let scaled = entity::resolve_checked(
        &[
            Mention::new("Alice Smith", vec![1e308, 1e308]),
            Mention::new("Alice", vec![1e-300, 1e-300]),
        ],
        EntityType::Character,
    )
    .unwrap();
    assert_eq!(scaled, simple);
    assert!(
        entity::resolve_checked(
            &[Mention::new("Alice", vec![f64::NAN])],
            EntityType::Character
        )
        .is_err()
    );
    assert!(
        entity::resolve_checked(
            &[Mention::new("Alice", vec![f64::INFINITY])],
            EntityType::Character
        )
        .is_err()
    );
    assert!(
        entity::resolve_checked(
            &[
                Mention::new("Alice", vec![1.0]),
                Mention::new("Alice", vec![1.0, 0.0])
            ],
            EntityType::Character
        )
        .is_err()
    );
}

#[test]
fn automatic_ids_are_namespace_unique_and_homonyms_survive_unrelated_corrections() {
    let unicode = entity::resolve_checked(
        &[Mention::new("李", vec![]), Mention::new("王", vec![])],
        EntityType::Character,
    )
    .unwrap();
    assert_eq!(unicode.len(), 2);
    assert_ne!(unicode[0].id, unicode[1].id);
    let symbols = entity::resolve_checked(
        &[Mention::new("☀", vec![]), Mention::new("☂", vec![])],
        EntityType::Symbol,
    )
    .unwrap();
    assert_eq!(symbols.len(), 2);
    assert_ne!(symbols[0].id, symbols[1].id);
    let character =
        entity::resolve_checked(&[Mention::new("Alice", vec![])], EntityType::Character).unwrap();
    let setting =
        entity::resolve_checked(&[Mention::new("Alice", vec![])], EntityType::Setting).unwrap();
    assert_ne!(character[0].id, setting[0].id);
    let mut mentions = vec![
        Mention::new("John Smith", vec![1.0, 0.0]),
        Mention::new("John Smith", vec![0.0, 1.0]),
    ];
    let before = entity::resolve_checked(&mentions, EntityType::Character).unwrap();
    assert_eq!(before.len(), 2);
    assert_ne!(before[0].id, before[1].id);
    let corrections = [EntityCorrection {
        surface: "Absent".into(),
        action: CorrectionAction::Reject,
    }];
    assert_eq!(
        entity::resolve_with_corrections_checked(&mentions, EntityType::Character, &corrections)
            .unwrap(),
        before
    );
    assert_eq!(
        entity::resolve_with_corrections_checked(&mentions, EntityType::Character, &[]).unwrap(),
        before
    );
    mentions.reverse();
    assert_eq!(
        entity::resolve_checked(&mentions, EntityType::Character).unwrap(),
        before
    );
}

#[test]
fn canonical_ids_use_specific_forms_and_corrections_retain_authority_and_exact_counts() {
    let mut mentions = vec![
        Mention::new("Robert Smith", vec![]),
        Mention::new("Bob", vec![]),
    ];
    let before = entity::resolve_checked(&mentions, EntityType::Character).unwrap();
    mentions.extend((0..5).map(|_| Mention::new("Bob", vec![])));
    let after = entity::resolve_checked(&mentions, EntityType::Character).unwrap();
    assert_eq!(before[0].id, after[0].id);
    assert_eq!(after[0].canonical, "Bob");
    let corrections = [
        EntityCorrection {
            surface: "Bob".into(),
            action: CorrectionAction::Reject,
        },
        EntityCorrection {
            surface: "Bob".into(),
            action: CorrectionAction::Assign {
                canonical: "Author identity".into(),
            },
        },
    ];
    let corrected =
        entity::resolve_with_corrections_checked(&mentions, EntityType::Character, &corrections)
            .unwrap();
    assert_eq!(
        corrected
            .iter()
            .map(|entity| entity.mention_count)
            .sum::<usize>(),
        7
    );
    assert_eq!(
        corrected
            .iter()
            .find(|entity| entity.canonical == "Author identity")
            .unwrap()
            .mention_count,
        6
    );
    assert!(
        entity::resolve_with_corrections_checked(
            &mentions,
            EntityType::Character,
            &[EntityCorrection {
                surface: "Bob".into(),
                action: CorrectionAction::Assign {
                    canonical: "".into()
                }
            }]
        )
        .is_err()
    );
    for value in [
        serde_json::json!({"kind":"reject","canonical":"Alice"}),
        serde_json::json!({"kind":"assign","canonical":"Alice","unknown":true}),
    ] {
        assert!(serde_json::from_value::<CorrectionAction>(value).is_err());
    }
}

#[test]
fn entity_count_surface_and_context_budgets_have_exact_boundaries() {
    for count in [
        entity::MAX_ENTITY_MENTIONS - 1,
        entity::MAX_ENTITY_MENTIONS,
        entity::MAX_ENTITY_MENTIONS + 1,
    ] {
        let mentions: Vec<_> = (0..count)
            .map(|i| Mention::new(format!("Entity{i}"), vec![]))
            .collect();
        let result = entity::resolve_checked(&mentions, EntityType::Character);
        assert_eq!(result.is_ok(), count <= entity::MAX_ENTITY_MENTIONS);
        if let Ok(entities) = result {
            assert_eq!(entities.len(), count);
            assert_eq!(
                entities
                    .iter()
                    .map(|entity| &entity.id)
                    .collect::<BTreeSet<_>>()
                    .len(),
                count
            );
        }
    }
    for length in [255, 256, 257] {
        assert_eq!(
            entity::resolve_checked(
                &[Mention::new("x".repeat(length), vec![])],
                EntityType::Character
            )
            .is_ok(),
            length <= 256
        );
    }
    for length in [4095, 4096, 4097] {
        assert_eq!(
            entity::resolve_checked(
                &[Mention::new("Alice", vec![1.0; length])],
                EntityType::Character
            )
            .is_ok(),
            length <= 4096
        );
    }
    for count in [
        entity::MAX_ENTITY_CONTEXT_VALUES - 1,
        entity::MAX_ENTITY_CONTEXT_VALUES,
        entity::MAX_ENTITY_CONTEXT_VALUES + 1,
    ] {
        let mut remaining = count;
        let mut mentions = Vec::new();
        while remaining >= 128 {
            mentions.push(Mention::new("Alice", vec![1.0; 128]));
            remaining -= 128;
        }
        if remaining > 0 {
            mentions.push(Mention::new("Alice", vec![0.0; remaining]));
        }
        assert_eq!(
            entity::resolve_checked(&mentions, EntityType::Character).is_ok(),
            count <= entity::MAX_ENTITY_CONTEXT_VALUES
        );
    }
}
