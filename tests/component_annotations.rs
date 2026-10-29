use narrative_lens::{
    craft::subtext::*,
    manuscript::{Manuscript, Scene},
    structure::{author_events::AuthorSource, thematic_argument::*},
    substrate::utils::AnalysisAvailability,
};

fn manuscript() -> Manuscript {
    Manuscript {
        document_id: "annotations".into(),
        scenes: vec![
            Scene::new("a", "🜂abcdef I am fine."),
            Scene::new("b", "Alice reveals the hidden meaning."),
        ],
        known_characters: vec!["Alice".into(), "Bob".into()],
        configuration: Default::default(),
    }
}
fn source(m: &Manuscript, id: &str, start: usize, end: usize) -> AuthorSource {
    let text = &m.scenes.iter().find(|scene| scene.id == id).unwrap().text;
    AuthorSource {
        scene_id: id.into(),
        utf16_start: text[..start].encode_utf16().count() as u32,
        utf16_end: text[..end].encode_utf16().count() as u32,
        quote: text[start..end].into(),
    }
}
fn subtext(m: &Manuscript, id: &str, start: usize, end: usize) -> AuthorSubtextAnnotation {
    AuthorSubtextAnnotation {
        id: id.into(),
        surface: source(m, "a", start, end),
        implied: "An author-declared interpretation".into(),
        subtext_type: SubtextType::VerbalIrony,
        characters: vec!["alice".into()],
        resolution: None,
    }
}
fn thesis(id: &str, opposes: &[&str]) -> AuthorThematicThesis {
    AuthorThematicThesis {
        id: id.into(),
        statement: format!("An author-declared thesis {id}"),
        opposes: opposes.iter().map(|id| (*id).into()).collect(),
    }
}
fn thematic(m: &Manuscript, id: &str, t: &str, scene: &str) -> AuthorThematicAnnotation {
    AuthorThematicAnnotation {
        id: id.into(),
        thesis_id: t.into(),
        source: source(m, scene, 0, 4),
        valence: EvidenceValence::Supporting,
        delivery: EvidenceDelivery::Dialogue {
            speaker: "alice".into(),
        },
    }
}

#[test]
fn subtext_overlaps_use_union_coverage_and_explicit_resolution_links() {
    let m = manuscript();
    let a = subtext(&m, "first", 0, 7);
    let mut b = subtext(&m, "second", 4, 10);
    b.resolution = Some(source(&m, "b", 0, 5));
    let result = analyze_subtext_annotations(&m, &[b, a]).unwrap();
    assert_eq!(
        result
            .instances
            .iter()
            .map(|i| i.id.as_str())
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
    assert_eq!(result.scene_coverage[0].annotation_count, 2);
    assert_eq!(result.scene_coverage[0].covered_utf16, 8);
    assert_eq!(result.scene_coverage[1].covered_utf16, 0);
    assert_eq!(result.resolved_count, 1);
    assert_eq!(result.unresolved_ids, ["first"]);
    assert_eq!(result.character_counts["Alice"], 2);
    assert_eq!(result.type_counts["VerbalIrony"], 2);
    assert!(result.instances[1].resolution.is_some());
    assert_eq!(result.availability, AnalysisAvailability::Complete);
    assert_eq!(
        analyze_subtext(&[&m.scenes[0].text], &[], &[], "literary").availability,
        AnalysisAvailability::Unavailable
    );
}

#[test]
fn evidence_ids_and_interpretations_survive_reorder_without_chronology_inference() {
    let mut m = manuscript();
    let mut a = subtext(&m, "first", 0, 7);
    a.resolution = Some(source(&m, "b", 0, 5));
    let before = analyze_subtext_annotations(&m, &[a.clone()]).unwrap();
    m.scenes.reverse();
    let after = analyze_subtext_annotations(&m, &[a]).unwrap();
    assert_eq!(
        before.instances[0].surface.scene_id,
        after.instances[0].surface.scene_id
    );
    assert_eq!(
        before.instances[0].surface.range,
        after.instances[0].surface.range
    );
    assert_eq!(before.resolved_count, after.resolved_count);
    assert_eq!(after.scene_coverage[0].scene_id, "b");
    assert_eq!(
        after.instances[0].resolution.as_ref().unwrap().scene_id,
        "b"
    );
}

#[test]
fn subtext_rejects_duplicate_ids_stale_quotes_surrogate_splits_and_unknown_actors() {
    let m = manuscript();
    let a = subtext(&m, "first", 0, 4);
    assert!(analyze_subtext_annotations(&m, &[a.clone(), a.clone()]).is_err());
    let mut bad = a.clone();
    bad.surface.quote = "invented".into();
    assert!(analyze_subtext_annotations(&m, &[bad]).is_err());
    let mut bad = a.clone();
    bad.surface.utf16_start = 1;
    assert!(analyze_subtext_annotations(&m, &[bad]).is_err());
    let mut bad = a;
    bad.characters = vec!["Charlie".into()];
    assert!(analyze_subtext_annotations(&m, &[bad]).is_err());
}

#[test]
fn thematic_links_and_counts_are_author_supplied_not_quality_scores() {
    let m = manuscript();
    let theses = [
        thesis("hope", &["fear", "fear"]),
        thesis("fear", &["hope"]),
        thesis("unassessed", &[]),
    ];
    let mut e = thematic(&m, "e1", "hope", "a");
    e.delivery = EvidenceDelivery::CharacterAction {
        character: "alice".into(),
        decision: "stay".into(),
    };
    let mut c = thematic(&m, "e2", "hope", "b");
    c.valence = EvidenceValence::Contradicting;
    let result = analyze_thematic_annotations(&m, &theses, &[c, e]).unwrap();
    assert_eq!(result.oppositions, [("fear".into(), "hope".into())]);
    let hope = result
        .summaries
        .iter()
        .find(|s| s.thesis_id == "hope")
        .unwrap();
    assert_eq!(hope.evidence_count, 2);
    assert_eq!(hope.valence_counts["Supporting"], 1);
    assert_eq!(hope.valence_counts["Contradicting"], 1);
    assert_eq!(hope.character_counts["Alice"], 2);
    assert_eq!(hope.scene_ids, ["a", "b"]);
    assert_eq!(result.unassessed_thesis_ids, ["fear", "unassessed"]);
    let encoded = serde_json::to_value(result).unwrap();
    assert!(encoded.get("argumentStrength").is_none());
    assert!(encoded.get("propagandaRisk").is_none());
    assert_eq!(
        track_thematic_argument(&["A passage"], &[], &[], &[], "literary").availability,
        AnalysisAvailability::Unavailable
    );
}

#[test]
fn thematic_unknown_self_duplicate_and_legacy_index_links_are_rejected() {
    let m = manuscript();
    for theses in [
        vec![thesis("a", &["missing"])],
        vec![thesis("a", &["a"])],
        vec![thesis("a", &[]), thesis("a", &[])],
    ] {
        assert!(analyze_thematic_annotations(&m, &theses, &[]).is_err());
    }
    let t = [thesis("t", &[])];
    assert!(analyze_thematic_annotations(&m, &t, &[thematic(&m, "e", "missing", "a")]).is_err());
    let mut e = thematic(&m, "e", "t", "a");
    e.delivery = EvidenceDelivery::StructuralParallel { parallel_scene: 1 };
    assert!(analyze_thematic_annotations(&m, &t, &[e.clone()]).is_err());
    for scene in ["missing", "a"] {
        e.delivery = EvidenceDelivery::StructuralParallelById {
            scene_id: scene.into(),
        };
        assert!(analyze_thematic_annotations(&m, &t, &[e.clone()]).is_err());
    }
}

#[test]
fn thematic_scene_reorder_preserves_stable_parallel_links_and_counts() {
    let mut m = manuscript();
    let t = [thesis("t", &[])];
    let mut e = thematic(&m, "e", "t", "a");
    e.delivery = EvidenceDelivery::StructuralParallelById {
        scene_id: "b".into(),
    };
    let before = analyze_thematic_annotations(&m, &t, &[e.clone()]).unwrap();
    m.scenes.reverse();
    let after = analyze_thematic_annotations(&m, &t, &[e]).unwrap();
    assert_eq!(
        serde_json::to_value(&before.evidence).unwrap(),
        serde_json::to_value(&after.evidence).unwrap()
    );
    assert_eq!(
        before.summaries[0].delivery_counts,
        after.summaries[0].delivery_counts
    );
    assert_eq!(after.scene_coverage[0].scene_id, "b");
}

#[test]
fn annotation_permutation_and_empty_inputs_are_deterministic() {
    let m = manuscript();
    let a = subtext(&m, "a1", 0, 4);
    let b = subtext(&m, "b1", 4, 7);
    assert_eq!(
        serde_json::to_value(analyze_subtext_annotations(&m, &[a.clone(), b.clone()]).unwrap())
            .unwrap(),
        serde_json::to_value(analyze_subtext_annotations(&m, &[b, a]).unwrap()).unwrap()
    );
    let empty = analyze_subtext_annotations(&m, &[]).unwrap();
    assert!(empty.instances.is_empty());
    assert!(empty.scene_coverage.iter().all(|s| s.covered_utf16 == 0));
    let empty = analyze_thematic_annotations(&m, &[], &[]).unwrap();
    assert!(empty.theses.is_empty());
    assert!(empty.evidence.is_empty());
    assert!(serde_json::from_str::<AuthorSubtextAnnotation>(r#"{"unknown":true}"#).is_err());
}

#[test]
fn annotation_payload_and_event_limits_cover_minus_equal_plus_one() {
    let m = manuscript();
    let template = subtext(&m, "first", 0, 4);
    for bytes in [4095, 4096, 4097] {
        let mut a = template.clone();
        a.implied = "x".repeat(bytes);
        assert_eq!(analyze_subtext_annotations(&m, &[a]).is_ok(), bytes <= 4096);
    }
    for count in [9999, 10000, 10001] {
        let annotations: Vec<_> = (0..count)
            .map(|i| {
                let mut a = template.clone();
                a.id = format!("a{i}");
                a.implied = "x".into();
                a.characters.clear();
                a
            })
            .collect();
        assert_eq!(
            analyze_subtext_annotations(&m, &annotations).is_ok(),
            count <= 10000
        );
    }
}

#[test]
fn metadata_budget_boundary_and_overlap_union_are_independent() {
    let mut m = manuscript();
    m.scenes = vec![Scene::new("a", "x".repeat(65536))];
    let mut prefix = Vec::new();
    let mut cost = 0;
    for i in 0..63 {
        let mut a = subtext(&m, &format!("a{i}"), 0, 65536);
        a.implied = "x".into();
        a.characters.clear();
        cost += a.id.len() + a.implied.len() + a.surface.scene_id.len() + a.surface.quote.len();
        prefix.push(a);
    }
    let mut last = subtext(&m, "last", 0, 1);
    last.implied = "x".into();
    last.characters.clear();
    let fixed = last.id.len() + last.implied.len() + last.surface.scene_id.len();
    let exact = 4 * 1024 * 1024 - cost - fixed;
    for change in [-1isize, 0, 1] {
        let bytes = (exact as isize + change) as usize;
        last.surface = source(&m, "a", 0, bytes);
        let mut annotations = prefix.clone();
        annotations.push(last.clone());
        let result = analyze_subtext_annotations(&m, &annotations);
        assert_eq!(result.is_ok(), change <= 0);
        if let Ok(result) = result {
            assert_eq!(result.scene_coverage[0].covered_utf16, 65536);
        }
    }
}

#[test]
fn thematic_annotation_permutation_and_combined_record_limit_are_explicit() {
    let m = manuscript();
    let t = [thesis("hope", &["fear"]), thesis("fear", &["hope"])];
    let a = thematic(&m, "a", "hope", "a");
    let b = thematic(&m, "b", "fear", "b");
    let first = analyze_thematic_annotations(&m, &t, &[a.clone(), b.clone()]).unwrap();
    let second = analyze_thematic_annotations(&m, &[t[1].clone(), t[0].clone()], &[b, a]).unwrap();
    assert_eq!(
        serde_json::to_value(first).unwrap(),
        serde_json::to_value(second).unwrap()
    );
    for count in [9999, 10000, 10001] {
        let t: Vec<_> = (0..count)
            .map(|i| AuthorThematicThesis {
                id: format!("t{i}"),
                statement: "x".into(),
                opposes: Vec::new(),
            })
            .collect();
        assert_eq!(
            analyze_thematic_annotations(&m, &t, &[]).is_ok(),
            count <= 10000
        );
    }
    for links in [9997, 9998, 9999] {
        let t = [thesis("a", &[]), thesis("b", &[])];
        let mut t = t;
        t[0].opposes = vec!["b".into(); links];
        let result = analyze_thematic_annotations(&m, &t, &[]);
        assert_eq!(result.is_ok(), links + t.len() <= 10000);
        if let Ok(result) = result {
            assert_eq!(result.oppositions, [("a".into(), "b".into())]);
        }
    }
}
