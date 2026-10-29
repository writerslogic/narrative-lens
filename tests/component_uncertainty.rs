use narrative_lens::{
    manuscript::{Manuscript, Scene},
    structure::narrative_entropy::*,
    substrate::utils::AnalysisAvailability,
};

fn manuscript() -> Manuscript {
    Manuscript {
        document_id: "uncertainty".into(),
        scenes: vec![
            Scene::new("a", "🜂Opening."),
            Scene::new("b", "Continuation."),
        ],
        known_characters: vec![],
        configuration: Default::default(),
    }
}
fn observation(question: &str, scene: &str, probabilities: &[f64]) -> SuppliedUncertainty {
    SuppliedUncertainty {
        question_id: question.into(),
        scene_id: scene.into(),
        outcomes: probabilities
            .iter()
            .enumerate()
            .map(|(i, p)| HypothesisProbability {
                id: format!("h{i}"),
                probability: *p,
            })
            .collect(),
    }
}

#[test]
fn shannon_known_values_and_declared_zero_outcomes_use_bits() {
    let input = [
        observation("fair", "a", &[0.5, 0.5]),
        observation("four", "a", &[0.25; 4]),
        observation("certain", "a", &[1.0, 0.0]),
        observation("zeros", "a", &[0.5, 0.5, 0.0, 0.0]),
        observation("single", "a", &[1.0]),
    ];
    let result = measure_narrative_uncertainty(&manuscript(), &input).unwrap();
    for m in result.measurements {
        let (bits, normalized, effective) = match m.question_id.as_str() {
            "fair" => (1.0, 1.0, 2.0),
            "four" => (2.0, 1.0, 4.0),
            "zeros" => (1.0, 0.5, 2.0),
            _ => (0.0, 0.0, 1.0),
        };
        assert!((m.entropy_bits - bits).abs() < 1e-12);
        assert!((m.normalized_entropy - normalized).abs() < 1e-12);
        assert!((m.effective_outcomes - effective).abs() < 1e-12);
        assert!(m.entropy_change_bits.is_none());
    }
    assert_eq!(result.availability, AnalysisAvailability::Complete);
}

#[test]
fn hypothesis_and_input_permutations_preserve_math_and_original_probabilities() {
    let m = manuscript();
    let a = observation("q", "a", &[0.1, 0.2, 0.7]);
    let b = observation("q", "b", &[0.2, 0.3, 0.5]);
    let first = measure_narrative_uncertainty(&m, &[a.clone(), b.clone()]).unwrap();
    let mut a2 = a;
    a2.outcomes.reverse();
    let mut b2 = b;
    b2.outcomes.rotate_left(1);
    let second = measure_narrative_uncertainty(&m, &[b2, a2]).unwrap();
    assert_eq!(
        serde_json::to_value(&first).unwrap(),
        serde_json::to_value(second).unwrap()
    );
    assert_eq!(first.measurements[0].outcomes[2].probability, 0.7);
    let change = first.measurements[1].entropy_bits - first.measurements[0].entropy_bits;
    assert_eq!(first.measurements[1].entropy_change_bits, Some(change));
    let mut reordered = m;
    reordered.scenes.reverse();
    let third = measure_narrative_uncertainty(
        &reordered,
        &[
            observation("q", "a", &[0.1, 0.2, 0.7]),
            observation("q", "b", &[0.2, 0.3, 0.5]),
        ],
    )
    .unwrap();
    assert_eq!(third.measurements[0].scene_id, "b");
    assert!((third.measurements[1].entropy_change_bits.unwrap() + change).abs() < 1e-12);
}

#[test]
fn invalid_distributions_ids_and_changed_hypotheses_fail() {
    let m = manuscript();
    for values in [
        vec![],
        vec![0.0, 0.0],
        vec![-f64::EPSILON, 1.0],
        vec![0.5, 0.4],
        vec![1.0 + f64::EPSILON],
        vec![f64::NAN, 1.0],
        vec![f64::INFINITY],
    ] {
        assert!(measure_narrative_uncertainty(&m, &[observation("q", "a", &values)]).is_err());
    }
    let a = observation("q", "a", &[0.5, 0.5]);
    let mut duplicate = a.clone();
    duplicate.outcomes[1].id = "h0".into();
    assert!(measure_narrative_uncertainty(&m, &[duplicate]).is_err());
    assert!(measure_narrative_uncertainty(&m, &[a.clone(), a.clone()]).is_err());
    assert!(measure_narrative_uncertainty(&m, &[observation("q", "missing", &[1.0])]).is_err());
    let mut changed = observation("q", "b", &[0.5, 0.5]);
    changed.outcomes[1].id = "different".into();
    assert!(measure_narrative_uncertainty(&m, &[a, changed]).is_err());
    let error =
        measure_narrative_uncertainty(&m, &[observation(&"private".repeat(1000), "a", &[1.0])])
            .unwrap_err()
            .to_string();
    assert!(!error.contains("private"));
    assert_eq!(
        measure_narrative_uncertainty(&m, &[]).unwrap().availability,
        AnalysisAvailability::Unavailable
    );
}

#[test]
fn roundoff_is_explicit_and_not_implicit_weight_normalization() {
    for delta in [-2e-9, -1e-9, -0.5e-9, 0.0, 0.5e-9, 1e-9, 2e-9] {
        let supplied = [0.5, 0.5 + delta];
        let result =
            measure_narrative_uncertainty(&manuscript(), &[observation("q", "a", &supplied)]);
        assert_eq!(result.is_ok(), delta.abs() <= 1e-9, "{delta}");
        if let Ok(result) = result {
            let m = &result.measurements[0];
            assert_eq!(m.outcomes[1].probability, supplied[1]);
            assert_eq!(m.probability_sum, supplied.iter().sum::<f64>());
            assert!((m.entropy_bits - 1.0).abs() < 1e-12);
        }
    }
    assert!(
        measure_narrative_uncertainty(&manuscript(), &[observation("q", "a", &[1.0, 1.0])])
            .is_err()
    );
}

#[test]
fn uncertainty_count_cell_id_and_metadata_budgets_cover_boundaries() {
    let m = manuscript();
    for count in [9999, 10000, 10001] {
        let input: Vec<_> = (0..count)
            .map(|i| observation(&format!("q{i}"), "a", &[1.0]))
            .collect();
        assert_eq!(
            measure_narrative_uncertainty(&m, &input).is_ok(),
            count <= 10000
        );
    }
    for count in [255, 256, 257] {
        assert_eq!(
            measure_narrative_uncertainty(
                &m,
                &[observation("q", "a", &vec![1.0 / count as f64; count])]
            )
            .is_ok(),
            count <= 256
        );
    }
    for count in [99999, 100000, 100001] {
        let mut input: Vec<_> = (0..1000)
            .map(|i| observation(&format!("q{i}"), "a", &vec![0.01; 100]))
            .collect();
        let last = count - 999 * 100;
        input[999] = observation("q999", "a", &vec![1.0 / last as f64; last]);
        assert_eq!(
            measure_narrative_uncertainty(&m, &input).is_ok(),
            count <= 100000
        );
    }
    for bytes in [255, 256, 257] {
        assert_eq!(
            measure_narrative_uncertainty(&m, &[observation(&"q".repeat(bytes), "a", &[1.0])])
                .is_ok(),
            bytes <= 256
        );
    }
    for target in [4 * 1024 * 1024 - 1, 4 * 1024 * 1024, 4 * 1024 * 1024 + 1] {
        let mut input: Vec<_> = (0..70)
            .map(|i| observation(&format!("q{i}"), "a", &[1.0 / 256.0; 256]))
            .collect();
        let mut bytes: usize = input
            .iter()
            .map(|o| {
                o.question_id.len()
                    + o.scene_id.len()
                    + o.outcomes.iter().map(|h| h.id.len()).sum::<usize>()
            })
            .sum();
        for o in &mut input {
            for h in &mut o.outcomes {
                let add = (target - bytes).min(256 - h.id.len());
                h.id.push_str(&"x".repeat(add));
                bytes += add;
            }
        }
        assert_eq!(bytes, target);
        assert_eq!(
            measure_narrative_uncertainty(&m, &input).is_ok(),
            target <= 4 * 1024 * 1024
        );
    }
}

fn dense_oracle(vectors: &[Vec<f64>], window: usize) -> Vec<Option<f64>> {
    let unit: Vec<_> = vectors
        .iter()
        .map(|v| {
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            (norm > 0.0).then(|| v.iter().map(|x| x / norm).collect::<Vec<_>>())
        })
        .collect();
    (0..vectors.len())
        .map(|i| {
            let query = unit[i].as_ref()?;
            let mut context = vec![0.0; query.len()];
            for v in unit[i.saturating_sub(window)..i].iter().flatten() {
                for (sum, x) in context.iter_mut().zip(v) {
                    *sum += x;
                }
            }
            let norm = context.iter().map(|x| x * x).sum::<f64>().sqrt();
            (norm > 1e-12)
                .then(|| 1.0 - query.iter().zip(&context).map(|(a, b)| a * b).sum::<f64>() / norm)
        })
        .collect()
}

#[test]
fn novelty_matches_independent_dense_math_over_bounded_windows() {
    let input: Vec<Vec<f64>> = (0..40)
        .map(|i| {
            (0..7)
                .map(|j| (((i * 17 + j * 13) % 31) as f64 - 15.0) / 9.0)
                .collect()
        })
        .collect();
    for window in [1, 2, 5, 40] {
        let options = SemanticNoveltyOptions {
            window,
            include_nearest: true,
        };
        let measured = measure_semantic_novelty(&input, &options).unwrap();
        for (point, expected) in measured.points.iter().zip(dense_oracle(&input, window)) {
            assert_eq!(point.novelty.is_some(), expected.is_some());
            if let Some(expected) = expected {
                assert!((point.novelty.unwrap() - expected).abs() < 1e-12);
            }
        }
        assert!(measured.points[0].novelty.is_none());
        assert_eq!(measured.availability, AnalysisAvailability::Complete);
        let again = measure_semantic_novelty(&input, &options).unwrap();
        assert_eq!(
            serde_json::to_value(measured).unwrap(),
            serde_json::to_value(again).unwrap()
        );
    }
}

#[test]
fn novelty_zero_cancellation_scale_and_nearest_ties_abstain_correctly() {
    let options = SemanticNoveltyOptions {
        window: 2,
        include_nearest: true,
    };
    let input = vec![
        vec![1.0, 0.0],
        vec![-1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.0, 0.0],
    ];
    let result = measure_semantic_novelty(&input, &options).unwrap();
    assert_eq!(result.points[1].novelty, Some(2.0));
    assert!(result.points[2].novelty.is_none());
    assert!(result.points[3].novelty.is_none());
    assert!(result.points[3].nearest_prior_scene.is_none());
    assert_eq!(result.points[2].nearest_prior_scene, Some(0));
    assert_eq!(result.availability, AnalysisAvailability::Partial);
    for scale in [1e-300, 1.0, 1e300] {
        let scaled: Vec<_> = input
            .iter()
            .map(|v| v.iter().map(|x| x * scale).collect())
            .collect();
        assert_eq!(
            serde_json::to_value(&result).unwrap(),
            serde_json::to_value(measure_semantic_novelty(&scaled, &options).unwrap()).unwrap()
        );
    }
    let unit = vec![vec![1.0, 1.0], vec![1e300, 1e300], vec![1e-300, 1e-300]];
    let tied = measure_semantic_novelty(&unit, &options).unwrap();
    assert_eq!(tied.points[2].nearest_prior_scene, Some(0));
    assert!(tied.points[2].novelty.unwrap() < 1e-12);
    let zeros = measure_semantic_novelty(&vec![vec![0.0; 2]; 4], &options).unwrap();
    assert_eq!(zeros.availability, AnalysisAvailability::Unavailable);
    assert!(zeros.mean_novelty.is_none());
}

#[test]
fn novelty_rejects_bad_shapes_nonfinite_and_resource_overflow_before_search() {
    let options = SemanticNoveltyOptions::default();
    for input in [
        vec![vec![]],
        vec![vec![1.0], vec![1.0, 2.0]],
        vec![vec![f64::NAN]],
        vec![vec![f64::INFINITY]],
    ] {
        assert!(measure_semantic_novelty(&input, &options).is_err());
    }
    assert_eq!(
        measure_semantic_novelty(&[], &options)
            .unwrap()
            .availability,
        AnalysisAvailability::Unavailable
    );
    for window in [0, 1, 9999, 10000, 10001] {
        assert_eq!(
            measure_semantic_novelty(
                &[],
                &SemanticNoveltyOptions {
                    window,
                    include_nearest: false
                }
            )
            .is_ok(),
            (1..=10000).contains(&window)
        );
    }
    for count in [9999, 10000, 10001] {
        assert_eq!(
            measure_semantic_novelty(&vec![vec![0.0]; count], &options).is_ok(),
            count <= 10000
        );
    }
    for dimensions in [4095, 4096, 4097] {
        assert_eq!(
            measure_semantic_novelty(&[vec![0.0; dimensions]], &options).is_ok(),
            dimensions <= 4096
        );
    }
    for rows in [2047, 2048, 2049] {
        assert_eq!(
            measure_semantic_novelty(&vec![vec![0.0; 4096]; rows], &options).is_ok(),
            rows <= 2048
        );
    }
    for rows in [1570, 1571, 1572] {
        let vectors = vec![vec![0.0; 4000]; rows];
        let result = measure_semantic_novelty(
            &vectors,
            &SemanticNoveltyOptions {
                window: 16,
                include_nearest: true,
            },
        );
        assert_eq!(result.is_ok(), rows <= 1571);
    }
}
