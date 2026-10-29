use narrative_lens::craft::lexical::{reciprocal_rank_fusion, reciprocal_rank_fusion_checked};

fn ranking(ids: &[&str]) -> Vec<String> {
    ids.iter().map(|id| (*id).into()).collect()
}

#[test]
fn duplicates_within_one_source_do_not_add_votes_or_shift_unique_ranks() {
    let original = vec![ranking(&["a", "b"]), ranking(&["b", "a"])];
    let duplicates = vec![ranking(&["a", "a", "b"]), ranking(&["b", "b", "a"])];
    let expected = reciprocal_rank_fusion_checked(&original, 0.0, 10).unwrap();
    assert_eq!(expected, vec![("a".into(), 1.5), ("b".into(), 1.5)]);
    assert_eq!(
        reciprocal_rank_fusion_checked(&duplicates, 0.0, 10).unwrap(),
        expected
    );
    let single = reciprocal_rank_fusion_checked(&[ranking(&["a", "b"])], 60.0, 10).unwrap();
    let two_sources =
        reciprocal_rank_fusion_checked(&[ranking(&["a", "b"]), ranking(&["a", "b"])], 60.0, 10)
            .unwrap();
    assert_eq!(two_sources[0].1, single[0].1 * 2.0);
}

#[test]
fn damping_and_resource_boundaries_reject_nonfinite_or_unbounded_work() {
    for k in [
        -f64::EPSILON,
        0.0,
        f64::EPSILON,
        f64::MAX,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let result = reciprocal_rank_fusion_checked(&[ranking(&["a"])], k, 1);
        assert_eq!(result.is_ok(), k.is_finite() && k >= 0.0);
        if let Ok(rows) = result {
            assert!(rows.iter().all(|(_, score)| score.is_finite()));
        }
    }
    assert!(reciprocal_rank_fusion(&[ranking(&["a"])], -1.0, 10).is_empty());
    for count in [9_999, 10_000, 10_001] {
        assert_eq!(
            reciprocal_rank_fusion_checked(&vec![vec![]; count], 0.0, 1).is_ok(),
            count <= 10_000
        );
    }
    for count in [99_999, 100_000, 100_001] {
        assert_eq!(
            reciprocal_rank_fusion_checked(&[vec!["a".into(); count]], 0.0, 1).is_ok(),
            count <= 100_000
        );
    }
    for length in [255, 256, 257] {
        assert_eq!(
            reciprocal_rank_fusion_checked(&[vec!["a".repeat(length)]], 0.0, 1).is_ok(),
            length <= 256
        );
    }
}

#[test]
fn ties_and_truncation_are_deterministic_under_source_permutation() {
    let sources = vec![ranking(&["b"]), ranking(&["a"])];
    assert_eq!(
        reciprocal_rank_fusion_checked(&sources, 60.0, 1).unwrap()[0].0,
        "a"
    );
    let mut reversed = sources.clone();
    reversed.reverse();
    assert_eq!(
        reciprocal_rank_fusion_checked(&sources, 60.0, 10).unwrap(),
        reciprocal_rank_fusion_checked(&reversed, 60.0, 10).unwrap()
    );
    assert!(
        reciprocal_rank_fusion_checked(&sources, 60.0, 0)
            .unwrap()
            .is_empty()
    );
}
