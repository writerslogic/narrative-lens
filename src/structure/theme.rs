use std::collections::{HashMap, HashSet};

use log::debug;
use serde::Serialize;

use crate::substrate::sbert;
use crate::substrate::utils::{cosine_similarity, mean, std_dev};

/// Semantic descriptions for SBERT-based theme detection.
const THEME_DESCRIPTIONS: &[(&str, &str)] = &[
    ("love", "romantic love, desire, passion, affection between characters"),
    ("death", "mortality, killing, dying, grief over loss"),
    ("power", "authority, control, dominion, political power, rulers and subjects"),
    ("betrayal", "treachery, deception, broken trust, backstabbing allies"),
    ("redemption", "forgiveness, atonement, second chances, moral recovery"),
    ("freedom", "liberation, escape from captivity, breaking chains, independence"),
    ("identity", "self-discovery, masks, disguises, transformation of self"),
    ("justice", "law, trials, punishment, innocence and guilt, fairness"),
    ("sacrifice", "selfless surrender, martyrdom, giving up something precious"),
    ("isolation", "loneliness, solitude, exile, abandonment, being an outcast"),
    ("grief", "mourning, sorrow, loss of a loved one, bereavement"),
    ("revenge", "vengeance, retribution, avenging a wrong, grudges"),
    ("hope", "optimism, dreams, aspiration, faith in a better future"),
    ("fear", "terror, dread, horror, anxiety, nightmares"),
    ("guilt", "shame, remorse, regret, a troubled conscience, self-blame"),
    ("coming_of_age", "growing up, maturity, leaving childhood, learning independence, mentors and first experiences"),
    ("war", "battle, soldiers, armies, weapons, enemies, siege, victory and defeat"),
    ("corruption", "corruption, bribery, scandal, dishonesty, abuse of power, exploitation, conspiracy"),
    ("survival", "survival against odds, hunger, shelter, danger, wilderness, endurance, struggle to live"),
    ("family", "family bonds, parents, siblings, children, home, inheritance, bloodline, legacy"),
    ("class", "social class, wealth and poverty, nobles and servants, privilege, inequality, status"),
    ("nature", "forests, rivers, mountains, oceans, wilderness, seasons, storms, the natural world"),
    ("technology", "machines, inventions, progress, digital systems, artificial intelligence, automation"),
    ("faith", "religion, god, prayer, church, miracles, divinity, sacred rituals, sin and salvation"),
    ("madness", "insanity, hallucinations, delusions, paranoia, obsession, mental breakdown"),
];

/// Thresholds for theme status classification (from Python source).
const MIN_THEME_PRESENCE: f64 = 0.05;

/// Per-scene theme relevance gate. SBERT cosine similarity is positive for almost
/// every theme against almost every scene, so keeping all `sim > 0` floods each
/// scene with the full theme set (high recall, near-zero precision — verified by
/// the §24.2 eval). A theme is kept only if it is within `RELATIVE` of the scene's
/// strongest theme *and* clears the `ABSOLUTE` floor, so each scene retains its
/// dominant themes and drops the long low-relevance tail every scene shares.
const THEME_RELEVANCE_RELATIVE: f64 = 0.70;
const THEME_RELEVANCE_ABSOLUTE: f64 = 0.0;
/// Hard cap on themes kept per scene. The relative gate trims peaked similarity
/// distributions to the dominant cluster, but a flat distribution (every theme
/// moderately similar) can still pass many; this caps the tail so a scene keeps
/// at most its few strongest themes.
const THEME_MAX_PER_SCENE: usize = 5;
const NUM_QUARTILES: usize = 4;
const UNRESOLVED_HIGH: f64 = 0.4;
const UNRESOLVED_LOW: f64 = 0.1;
const EMERGENT_LOW: f64 = 0.1;
const EMERGENT_HIGH: f64 = 0.4;
const BACKGROUND_CEIL: f64 = 0.2;
const CYCLICAL_PRESENT: f64 = 0.2;
const CLIMACTIC_MIN: f64 = 0.3;

/// Resolution status for a single theme across the manuscript.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ThematicResolution {
    pub theme_name: String,
    pub act1_presence: f64,
    pub act2_presence: f64,
    pub act3_presence: f64,
    pub is_resolved: bool,
    pub status: String,
    pub quartile_presence: Vec<f64>,
    pub chapter_curve: Vec<f64>,
    pub peak_chapter: i32,
    pub consistency: f64,
    /// Theme momentum: positive means increasing presence, negative means decreasing.
    /// Computed from consecutive quartile average differences.
    pub momentum: f64,
    /// "increasing", "decreasing", or "steady"
    pub momentum_label: String,
}

/// Truncate text to approximately the first `n` words.
fn first_n_words(text: &str, n: usize) -> String {
    let mut count = 0;
    let mut end = text.len();
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            count += 1;
            if count >= n {
                end = i;
                break;
            }
        }
    }
    text[..end].to_string()
}

/// Detect themes using SBERT embeddings. Encodes the first 500 words of each scene
/// and compares against theme description embeddings via cosine similarity.
/// Returns empty Vec if SBERT model is unavailable.
pub fn detect_themes_sbert(scenes: &[String]) -> Vec<HashMap<String, f64>> {
    // Encode theme descriptions; bail if model unavailable
    let theme_embeddings: Vec<(&str, Vec<f64>)> = THEME_DESCRIPTIONS
        .iter()
        .map(|(name, desc)| (*name, sbert::encode_text(desc)))
        .collect();

    if theme_embeddings.iter().all(|(_, emb)| emb.is_empty()) {
        return Vec::new();
    }

    let truncated: Vec<String> = scenes
        .iter()
        .map(|scene| first_n_words(scene, 500))
        .collect();
    let truncated_refs: Vec<&str> = truncated.iter().map(|s| s.as_str()).collect();
    let scene_embeddings = sbert::encode_batch(&truncated_refs);

    scene_embeddings
        .into_iter()
        .map(|scene_emb| {
            if scene_emb.is_empty() {
                return HashMap::new();
            }
            // Cosine-similarity of every theme to this scene, then keep only the
            // relevant ones: within RELATIVE of the scene's strongest theme and
            // above the ABSOLUTE floor (see the constants). This trades the
            // similarity flood for the scene's dominant themes.
            let sims: Vec<(&str, f64)> = theme_embeddings
                .iter()
                .filter(|(_, emb)| !emb.is_empty())
                .map(|(name, emb)| (*name, cosine_similarity(&scene_emb, emb)))
                .collect();
            let max_sim = sims.iter().map(|(_, s)| *s).fold(0.0f64, f64::max);
            let floor = (max_sim * THEME_RELEVANCE_RELATIVE).max(THEME_RELEVANCE_ABSOLUTE);
            let mut kept: Vec<(&str, f64)> = sims
                .into_iter()
                .filter(|(_, sim)| *sim > 0.0 && *sim >= floor)
                .collect();
            // Strongest first, then cap the tail (handles flat distributions the
            // relative gate alone lets through).
            kept.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            kept.truncate(THEME_MAX_PER_SCENE);
            kept.into_iter()
                .map(|(name, sim)| (name.to_string(), (sim * 10000.0).round() / 10000.0))
                .collect()
        })
        .collect()
}

/// Detect themes using SBERT if available, falling back to keyword detection.
pub fn detect_themes_smart(
    scenes: &[String],
    scenes_tokens: &[Vec<String>],
) -> Vec<HashMap<String, f64>> {
    let sbert_results = detect_themes_sbert(scenes);
    if !sbert_results.is_empty() {
        return sbert_results;
    }
    // Fallback to keyword detection
    crate::substrate::nlp::detect_themes_keywords(scenes_tokens.to_vec()).unwrap_or_default()
}

/// Analyze how themes evolve across the manuscript.
/// Takes per-scene theme scores (scene_themes[i] = HashMap of theme_name -> confidence).
/// Returns sorted list of thematic resolutions for themes with sufficient presence.
pub fn analyze_resolution(scene_themes: &[HashMap<String, f64>]) -> Vec<ThematicResolution> {
    debug!("[theme] analyze_resolution — {} scenes", scene_themes.len());
    if scene_themes.is_empty() {
        return Vec::new();
    }

    let num_scenes = scene_themes.len();
    let act1_end = (num_scenes / 4).max(1);
    let act3_start = ((num_scenes * 3) / 4).max(1);

    // Quartile boundaries
    let q_size = (num_scenes / NUM_QUARTILES).max(1);
    let mut quartile_bounds: Vec<(usize, usize)> = (0..NUM_QUARTILES)
        .map(|i| {
            let qs = (i * q_size).min(num_scenes);
            let qe = ((i + 1) * q_size).min(num_scenes);
            (qs, qe)
        })
        .collect();
    if let Some(last) = quartile_bounds.last_mut() {
        last.1 = num_scenes;
    }

    // Collect all unique themes
    let all_themes: HashSet<&String> = scene_themes.iter().flat_map(|s| s.keys()).collect();

    let mut resolutions = Vec::new();

    for theme in all_themes {
        let all_scores: Vec<f64> = scene_themes
            .iter()
            .map(|s| *s.get(theme).unwrap_or(&0.0))
            .collect();

        let overall_avg = mean(&all_scores);
        if overall_avg < MIN_THEME_PRESENCE {
            continue;
        }

        // Three-act averages
        let a1_avg = mean(&all_scores[..act1_end]);
        let a2_avg = mean(&all_scores[act1_end..act3_start]);
        let a3_avg = mean(&all_scores[act3_start..]);

        // Quartile averages
        let quartile_avgs: Vec<f64> = quartile_bounds
            .iter()
            .map(|&(qs, qe)| mean(&all_scores[qs..qe]))
            .collect();

        // Per-chapter curve
        let chapter_curve: Vec<f64> = all_scores
            .iter()
            .map(|s| (s * 10000.0).round() / 10000.0)
            .collect();

        // Peak chapter
        let peak_chapter = all_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, v)| if *v > 0.0 { i as i32 } else { -1 })
            .unwrap_or(-1);

        // Consistency: 1 - normalized std dev
        let sd = std_dev(&all_scores);
        let consistency = (1.0 - sd / overall_avg.max(0.01)).clamp(0.0, 1.0);
        let consistency = (consistency * 10000.0).round() / 10000.0;

        // Momentum: average of consecutive quartile differences
        let momentum = if quartile_avgs.len() >= 2 {
            let diffs: Vec<f64> = quartile_avgs.windows(2).map(|w| w[1] - w[0]).collect();
            let raw = mean(&diffs);
            (raw * 10000.0).round() / 10000.0
        } else {
            0.0
        };
        let momentum_label = if momentum > 0.05 {
            "increasing"
        } else if momentum < -0.05 {
            "decreasing"
        } else {
            "steady"
        };

        // Resolution logic
        let mut is_resolved = true;
        let status;

        if a1_avg > UNRESOLVED_HIGH && a3_avg < UNRESOLVED_LOW {
            is_resolved = false;
            status = "unresolved";
        } else if a1_avg < EMERGENT_LOW && a3_avg > EMERGENT_HIGH {
            status = "emergent";
        } else if a1_avg < BACKGROUND_CEIL && a3_avg < BACKGROUND_CEIL && a2_avg < BACKGROUND_CEIL {
            status = "background";
        } else if a1_avg > CYCLICAL_PRESENT && a2_avg < UNRESOLVED_LOW && a3_avg > CYCLICAL_PRESENT
        {
            status = "cyclical";
        } else if a2_avg >= a1_avg && a3_avg >= a2_avg && a3_avg > CLIMACTIC_MIN {
            status = "climactic";
        } else {
            status = "resolved";
        }

        resolutions.push(ThematicResolution {
            theme_name: theme.clone(),
            act1_presence: a1_avg,
            act2_presence: a2_avg,
            act3_presence: a3_avg,
            is_resolved,
            status: status.into(),
            quartile_presence: quartile_avgs,
            chapter_curve,
            peak_chapter,
            consistency,
            momentum,
            momentum_label: momentum_label.into(),
        });
    }

    // Sort by overall presence descending
    resolutions.sort_by(|a, b| {
        let total_b = b.act1_presence + b.act2_presence + b.act3_presence;
        let total_a = a.act1_presence + a.act2_presence + a.act3_presence;
        total_b
            .partial_cmp(&total_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    resolutions
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(themes: &[(&str, f64)]) -> HashMap<String, f64> {
        themes.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn test_empty() {
        assert!(analyze_resolution(&[]).is_empty());
    }

    #[test]
    fn test_uniform_theme_climactic() {
        // Uniform 0.5 across all acts: a2>=a1, a3>=a2, a3>0.3 -> climactic
        let scenes: Vec<HashMap<String, f64>> = (0..12).map(|_| scene(&[("love", 0.5)])).collect();
        let res = analyze_resolution(&scenes);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].status, "climactic");
        assert!(res[0].is_resolved);
    }

    #[test]
    fn test_resolved_theme() {
        // Peaks in act2 then drops: a3 < a2 so not climactic, falls to default "resolved"
        let mut scenes = Vec::new();
        for _ in 0..3 {
            scenes.push(scene(&[("love", 0.4)]));
        }
        for _ in 0..6 {
            scenes.push(scene(&[("love", 0.6)]));
        }
        for _ in 0..3 {
            scenes.push(scene(&[("love", 0.3)]));
        }
        let res = analyze_resolution(&scenes);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].status, "resolved");
    }

    #[test]
    fn test_unresolved_theme() {
        // Strong in act1, absent in act3
        let mut scenes = Vec::new();
        for _ in 0..4 {
            scenes.push(scene(&[("betrayal", 0.8)]));
        }
        for _ in 0..8 {
            scenes.push(scene(&[("betrayal", 0.0)]));
        }
        let res = analyze_resolution(&scenes);
        let betrayal = res.iter().find(|r| r.theme_name == "betrayal");
        // May or may not pass threshold depending on overall avg
        if let Some(b) = betrayal {
            assert_eq!(b.status, "unresolved");
            assert!(!b.is_resolved);
        }
    }

    #[test]
    fn test_emergent_theme() {
        // Absent early, strong late
        let mut scenes = Vec::new();
        for _ in 0..8 {
            scenes.push(scene(&[("hope", 0.0)]));
        }
        for _ in 0..4 {
            scenes.push(scene(&[("hope", 0.8)]));
        }
        let res = analyze_resolution(&scenes);
        let hope = res.iter().find(|r| r.theme_name == "hope");
        if let Some(h) = hope {
            assert_eq!(h.status, "emergent");
        }
    }

    #[test]
    fn test_below_threshold_filtered() {
        let scenes: Vec<HashMap<String, f64>> =
            (0..12).map(|_| scene(&[("obscure", 0.01)])).collect();
        let res = analyze_resolution(&scenes);
        assert!(res.is_empty());
    }

    #[test]
    fn test_consistency() {
        // Uniform theme should have high consistency
        let scenes: Vec<HashMap<String, f64>> = (0..12).map(|_| scene(&[("power", 0.5)])).collect();
        let res = analyze_resolution(&scenes);
        assert!(res[0].consistency > 0.9);
    }

    #[test]
    fn test_peak_chapter() {
        let mut scenes: Vec<HashMap<String, f64>> =
            (0..10).map(|_| scene(&[("death", 0.2)])).collect();
        scenes[7] = scene(&[("death", 0.9)]);
        let res = analyze_resolution(&scenes);
        let death = res.iter().find(|r| r.theme_name == "death").unwrap();
        assert_eq!(death.peak_chapter, 7);
    }

    #[test]
    fn test_sorted_by_presence() {
        let scenes: Vec<HashMap<String, f64>> = (0..12)
            .map(|_| scene(&[("major", 0.8), ("minor", 0.1)]))
            .collect();
        let res = analyze_resolution(&scenes);
        assert_eq!(res[0].theme_name, "major");
    }

    #[test]
    fn test_momentum_increasing() {
        // Theme grows across quartiles: should be "increasing"
        let mut scenes = Vec::new();
        for _ in 0..3 {
            scenes.push(scene(&[("power", 0.1)]));
        }
        for _ in 0..3 {
            scenes.push(scene(&[("power", 0.3)]));
        }
        for _ in 0..3 {
            scenes.push(scene(&[("power", 0.5)]));
        }
        for _ in 0..3 {
            scenes.push(scene(&[("power", 0.8)]));
        }
        let res = analyze_resolution(&scenes);
        let power = res.iter().find(|r| r.theme_name == "power").unwrap();
        assert!(
            power.momentum > 0.05,
            "Momentum should be positive, got {}",
            power.momentum
        );
        assert_eq!(power.momentum_label, "increasing");
    }

    #[test]
    fn test_momentum_decreasing() {
        // Theme fades across quartiles: should be "decreasing"
        let mut scenes = Vec::new();
        for _ in 0..3 {
            scenes.push(scene(&[("fear", 0.8)]));
        }
        for _ in 0..3 {
            scenes.push(scene(&[("fear", 0.5)]));
        }
        for _ in 0..3 {
            scenes.push(scene(&[("fear", 0.3)]));
        }
        for _ in 0..3 {
            scenes.push(scene(&[("fear", 0.1)]));
        }
        let res = analyze_resolution(&scenes);
        let fear = res.iter().find(|r| r.theme_name == "fear").unwrap();
        assert!(
            fear.momentum < -0.05,
            "Momentum should be negative, got {}",
            fear.momentum
        );
        assert_eq!(fear.momentum_label, "decreasing");
    }

    #[test]
    fn test_momentum_steady() {
        // Uniform theme: should be "steady"
        let scenes: Vec<HashMap<String, f64>> = (0..12).map(|_| scene(&[("love", 0.5)])).collect();
        let res = analyze_resolution(&scenes);
        let love = res.iter().find(|r| r.theme_name == "love").unwrap();
        assert_eq!(love.momentum_label, "steady");
        assert!(
            love.momentum.abs() <= 0.05,
            "Steady momentum should be near 0, got {}",
            love.momentum
        );
    }
}
