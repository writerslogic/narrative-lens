use narrative_lens::{
    manuscript::{Configuration, Manuscript, Scene},
    structure::author_events::*,
};

fn manuscript() -> Manuscript {
    Manuscript {
        document_id: "mystery".into(),
        scenes: vec![
            Scene::new("question", "😀 Who stole the silver key?"),
            Scene::new("answer", "Alice stole the silver key."),
        ],
        known_characters: vec![],
        configuration: Configuration::default(),
    }
}

fn source(scene: &str, text: &str, start: u32) -> AuthorSource {
    AuthorSource {
        scene_id: scene.into(),
        utf16_start: start,
        utf16_end: start + text.encode_utf16().count() as u32,
        quote: text.into(),
    }
}

fn events() -> Vec<AuthorPromiseEvent> {
    vec![
        AuthorPromiseEvent::Setup {
            id: "missing-key".into(),
            description: "Identity of the thief".into(),
            evidence: source("question", "Who stole the silver key?", 3),
        },
        AuthorPromiseEvent::Resolution {
            id: "thief-revealed".into(),
            setup_id: "missing-key".into(),
            outcome: AuthorResolution::Fulfilled,
            evidence: source("answer", "Alice stole the silver key.", 0),
        },
    ]
}

#[test]
fn answered_question_requires_an_explicit_link_at_the_actual_source() {
    let input = manuscript();
    let events = events();
    let unknown = analyze_promise_events(&input, &events[..1]).unwrap();
    assert_eq!(unknown.unknown_count, 1);
    assert_eq!(unknown.fulfilled_count, 0);
    assert!(unknown.promises[0].resolution.is_none());
    let ledger = analyze_promise_events(&input, &events).unwrap();
    assert_eq!(ledger.fulfilled_count, 1);
    assert_eq!(ledger.unknown_count, 0);
    let promise = &ledger.promises[0];
    assert_eq!(promise.setup.range.byte_start, 5);
    let resolution = promise.resolution.as_ref().unwrap();
    assert_eq!(resolution.evidence.scene_id, "answer");
    assert_eq!(resolution.event_id, "thief-revealed");
    for evidence in [&promise.setup, &resolution.evidence] {
        let text = &input
            .scenes
            .iter()
            .find(|s| s.id == evidence.scene_id)
            .unwrap()
            .text;
        assert_eq!(
            text.get(evidence.range.byte_start..evidence.range.byte_end),
            Some(evidence.quote.as_str())
        );
        let utf16: Vec<_> = text.encode_utf16().collect();
        assert_eq!(
            String::from_utf16(
                &utf16[evidence.range.utf16_start as usize..evidence.range.utf16_end as usize]
            )
            .unwrap(),
            evidence.quote
        );
    }
}

#[test]
fn identities_survive_reorder_and_source_edits_require_refreshed_ranges() {
    let mut input = manuscript();
    let mut events = events();
    let before = analyze_promise_events(&input, &events).unwrap();
    input.scenes.reverse();
    events.reverse();
    assert_eq!(analyze_promise_events(&input, &events).unwrap(), before);
    input.scenes[0].text.insert_str(0, "Later: ");
    assert!(analyze_promise_events(&input, &events).is_err());
    if let AuthorPromiseEvent::Resolution { evidence, .. } = &mut events[0] {
        evidence.utf16_start += 7;
        evidence.utf16_end += 7;
    }
    let after = analyze_promise_events(&input, &events).unwrap();
    assert_eq!(after.promises[0].id, before.promises[0].id);
    assert_eq!(
        after.promises[0].resolution.as_ref().unwrap().event_id,
        "thief-revealed"
    );
}

#[test]
fn dangling_duplicate_and_invalid_source_annotations_fail_atomically() {
    let input = manuscript();
    let original = events();
    assert!(analyze_promise_events(&input, &original[1..]).is_err());
    let mut duplicate = original.clone();
    duplicate.push(original[0].clone());
    assert!(analyze_promise_events(&input, &duplicate).is_err());
    duplicate = original.clone();
    if let AuthorPromiseEvent::Resolution { id, .. } = &mut duplicate[1] {
        *id = "another-answer".into();
    }
    duplicate.push(original[1].clone());
    assert!(analyze_promise_events(&input, &duplicate).is_err());
    for bad in [1, 2, u32::MAX] {
        let mut invalid = original.clone();
        if let AuthorPromiseEvent::Setup { evidence, .. } = &mut invalid[0] {
            evidence.utf16_start = bad;
        }
        assert!(analyze_promise_events(&input, &invalid).is_err());
    }
    let mut unknown_scene = original.clone();
    if let AuthorPromiseEvent::Setup { evidence, .. } = &mut unknown_scene[0] {
        evidence.scene_id = "absent".into();
    }
    assert!(analyze_promise_events(&input, &unknown_scene).is_err());
    assert_eq!(
        analyze_promise_events(&input, &original)
            .unwrap()
            .fulfilled_count,
        1
    );
}

#[test]
fn distinct_author_outcomes_are_counted_without_quality_estimates() {
    let input = manuscript();
    for outcome in [
        AuthorResolution::Fulfilled,
        AuthorResolution::Partial,
        AuthorResolution::Subverted,
        AuthorResolution::Abandoned,
    ] {
        let mut events = events();
        if let AuthorPromiseEvent::Resolution {
            outcome: actual, ..
        } = &mut events[1]
        {
            *actual = outcome;
        }
        let ledger = analyze_promise_events(&input, &events).unwrap();
        assert_eq!(
            ledger.fulfilled_count
                + ledger.partial_count
                + ledger.subverted_count
                + ledger.abandoned_count,
            1
        );
        assert_eq!(
            ledger.fulfilled_count,
            usize::from(outcome == AuthorResolution::Fulfilled)
        );
        let json = serde_json::to_value(&ledger).unwrap();
        assert!(json["promises"][0]["resolution"].get("quality").is_none());
    }
}

#[test]
fn author_event_count_and_metadata_limits_have_defined_boundaries() {
    let input = manuscript();
    let setup = |id: String, description: String| AuthorPromiseEvent::Setup {
        id,
        description,
        evidence: source("question", "Who stole the silver key?", 3),
    };
    let mut many: Vec<_> = (0..=MAX_AUTHOR_EVENTS)
        .map(|i| setup(format!("setup-{i}"), "Question".into()))
        .collect();
    for count in [MAX_AUTHOR_EVENTS - 1, MAX_AUTHOR_EVENTS] {
        assert_eq!(
            analyze_promise_events(&input, &many[..count])
                .unwrap()
                .promises
                .len(),
            count
        );
    }
    assert!(analyze_promise_events(&input, &many).is_err());
    many.clear();
    for length in [255, 256, 257] {
        assert_eq!(
            analyze_promise_events(&input, &[setup("x".repeat(length), "Question".into())]).is_ok(),
            length <= 256
        );
    }
    for length in [4095, 4096, 4097] {
        assert_eq!(
            analyze_promise_events(&input, &[setup("question".into(), "x".repeat(length))]).is_ok(),
            length <= 4096
        );
    }
}

#[test]
fn quote_and_total_annotation_byte_budgets_are_enforced_at_the_boundary() {
    let mut input = manuscript();
    input.scenes = vec![Scene::new("s", "x".repeat(65_537))];
    for length in [65_535, 65_536, 65_537] {
        let event = AuthorPromiseEvent::Setup {
            id: "a".into(),
            description: "d".into(),
            evidence: source("s", &"x".repeat(length), 0),
        };
        assert_eq!(
            analyze_promise_events(&input, &[event]).is_ok(),
            length <= 65_536
        );
    }
    for limit in [
        MAX_AUTHOR_EVENT_BYTES - 1,
        MAX_AUTHOR_EVENT_BYTES,
        MAX_AUTHOR_EVENT_BYTES + 1,
    ] {
        let mut events = Vec::new();
        let mut remaining = limit;
        while remaining > 0 {
            let id = format!("event-{}", events.len());
            let overhead = id.len() + 2;
            let mut quote_bytes = (remaining - overhead).min(65_536);
            if remaining - overhead - quote_bytes < 32 && remaining - overhead - quote_bytes != 0 {
                quote_bytes -= 32;
            }
            events.push(AuthorPromiseEvent::Setup {
                id,
                description: "d".into(),
                evidence: source("s", &"x".repeat(quote_bytes), 0),
            });
            remaining -= overhead + quote_bytes;
        }
        assert_eq!(
            analyze_promise_events(&input, &events).is_ok(),
            limit <= MAX_AUTHOR_EVENT_BYTES
        );
    }
}
