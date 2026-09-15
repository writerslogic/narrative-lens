use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneConfidence {
    pub scene_id: usize,
    pub data_quality: f64,
    pub word_count_factor: f64,
    pub overall: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadabilityConfidence {
    pub fkgl_confidence: f64,
    pub overall: f64,
    pub sample_adequacy: f64,
}

/// Compute confidence for analysis of a scene based on its text content.
pub fn compute_scene_confidence(scene_text: &str, scene_id: usize) -> SceneConfidence {
    let word_count = scene_text.split_whitespace().count();

    let word_count_factor = if word_count >= 500 {
        1.0
    } else if word_count >= 200 {
        0.8
    } else if word_count >= 50 {
        0.6
    } else {
        0.3
    };

    let total_chars = scene_text.len();
    let data_quality = if total_chars == 0 {
        0.0
    } else {
        let analyzable = scene_text
            .chars()
            .filter(|c| !c.is_whitespace() && !c.is_ascii_punctuation())
            .count();
        analyzable as f64 / total_chars as f64
    };

    let overall = (word_count_factor * data_quality).sqrt();

    SceneConfidence {
        scene_id,
        data_quality,
        word_count_factor,
        overall,
    }
}

/// Compute confidence for a specific finding based on signal agreement and scene confidence.
pub fn compute_finding_confidence(
    signal_count: usize,
    total_signals: usize,
    scene_confidence: f64,
) -> f64 {
    if total_signals == 0 {
        return 0.0;
    }
    let agreement_ratio = signal_count as f64 / total_signals as f64;
    agreement_ratio * 0.7 + scene_confidence * 0.3
}

/// Compute confidence for readability metrics based on sample size.
pub fn compute_readability_confidence(
    word_count: usize,
    sentence_count: usize,
) -> ReadabilityConfidence {
    let word_adequacy = (word_count as f64 / 100.0).min(1.0);
    let sentence_adequacy = (sentence_count as f64 / 3.0).min(1.0);
    let sample_adequacy = word_adequacy * sentence_adequacy;

    let fkgl_confidence = sample_adequacy;

    ReadabilityConfidence {
        fkgl_confidence,
        overall: sample_adequacy,
        sample_adequacy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_confidence_short() {
        let conf = compute_scene_confidence("hello world", 0);
        assert_eq!(conf.scene_id, 0);
        assert_eq!(conf.word_count_factor, 0.3);
        assert!(conf.data_quality > 0.0);
        assert!(conf.overall < 1.0);
    }

    #[test]
    fn test_scene_confidence_empty() {
        let conf = compute_scene_confidence("", 5);
        assert_eq!(conf.data_quality, 0.0);
        assert_eq!(conf.overall, 0.0);
    }

    #[test]
    fn test_scene_confidence_long() {
        let text = "word ".repeat(600);
        let conf = compute_scene_confidence(&text, 1);
        assert_eq!(conf.word_count_factor, 1.0);
        assert!(conf.overall > 0.5);
    }

    #[test]
    fn test_scene_confidence_medium() {
        let text = "word ".repeat(100);
        let conf = compute_scene_confidence(&text, 2);
        assert_eq!(conf.word_count_factor, 0.6);
    }

    #[test]
    fn test_scene_confidence_200_words() {
        let text = "word ".repeat(250);
        let conf = compute_scene_confidence(&text, 3);
        assert_eq!(conf.word_count_factor, 0.8);
    }

    #[test]
    fn test_finding_confidence_zero_signals() {
        let result = compute_finding_confidence(0, 0, 0.8);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_finding_confidence_full_agreement() {
        let result = compute_finding_confidence(5, 5, 1.0);
        assert!((result - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_finding_confidence_partial() {
        let result = compute_finding_confidence(2, 4, 0.8);
        let expected = 0.5 * 0.7 + 0.8 * 0.3;
        assert!((result - expected).abs() < 1e-9);
    }

    #[test]
    fn test_readability_confidence_adequate() {
        let conf = compute_readability_confidence(200, 10);
        assert_eq!(conf.sample_adequacy, 1.0);
        assert_eq!(conf.fkgl_confidence, 1.0);
        assert_eq!(conf.overall, 1.0);
    }

    #[test]
    fn test_readability_confidence_low_words() {
        let conf = compute_readability_confidence(50, 10);
        assert!((conf.sample_adequacy - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_readability_confidence_low_sentences() {
        let conf = compute_readability_confidence(200, 1);
        let expected = 1.0 * (1.0 / 3.0);
        assert!((conf.sample_adequacy - expected).abs() < 1e-9);
    }

    #[test]
    fn test_readability_confidence_zero() {
        let conf = compute_readability_confidence(0, 0);
        assert_eq!(conf.sample_adequacy, 0.0);
        assert_eq!(conf.overall, 0.0);
    }
}
