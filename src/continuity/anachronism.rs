use std::collections::HashMap;
use std::sync::OnceLock;

use serde::Serialize;

/// An anachronistic word found in the text.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnachronismItem {
    pub word: String,
    pub introduced_year: i32,
    pub target_year: i32,
    pub scene_id: usize,
}

const GRACE_PERIOD_YEARS: i32 = 20;

fn milestones() -> &'static HashMap<&'static str, i32> {
    static M: OnceLock<HashMap<&str, i32>> = OnceLock::new();
    M.get_or_init(|| {
        let mut m = HashMap::new();
        // Communication & media
        m.insert("internet", 1991);
        m.insert("email", 1972);
        m.insert("texting", 1992);
        m.insert("blog", 1997);
        m.insert("podcast", 2004);
        m.insert("selfie", 2002);
        m.insert("hashtag", 2007);
        m.insert("viral", 1990);
        m.insert("streaming", 1995);
        m.insert("wifi", 1997);
        m.insert("bluetooth", 1998);
        m.insert("emoji", 1999);
        m.insert("meme", 1976);
        m.insert("tweet", 2006);
        m.insert("google", 1998);
        // Technology
        m.insert("computer", 1940);
        m.insert("smartphone", 2007);
        m.insert("laptop", 1983);
        m.insert("software", 1958);
        m.insert("television", 1927);
        m.insert("radio", 1895);
        m.insert("telephone", 1876);
        m.insert("telegraph", 1837);
        m.insert("microchip", 1958);
        m.insert("transistor", 1947);
        m.insert("laser", 1960);
        m.insert("radar", 1935);
        m.insert("satellite", 1957);
        m.insert("robot", 1920);
        m.insert("algorithm", 1950);
        m.insert("database", 1960);
        m.insert("pixel", 1965);
        m.insert("digital", 1942);
        m.insert("analog", 1940);
        m.insert("cybernetic", 1948);
        // Electronics
        m.insert("semiconductor", 1950);
        m.insert("microprocessor", 1971);
        m.insert("silicon", 1824);
        m.insert("circuit", 1899);
        m.insert("diode", 1919);
        m.insert("capacitor", 1926);
        m.insert("resistor", 1909);
        m.insert("oscilloscope", 1897);
        m.insert("solenoid", 1823);
        // Transport
        m.insert("airplane", 1903);
        m.insert("helicopter", 1939);
        m.insert("automobile", 1886);
        m.insert("motorcycle", 1885);
        m.insert("subway", 1863);
        m.insert("escalator", 1897);
        m.insert("rocket", 1926);
        m.insert("spaceship", 1929);
        m.insert("jeep", 1940);
        m.insert("tank", 1916);
        m.insert("submarine", 1897);
        // Science & medicine
        m.insert("antibiotic", 1928);
        m.insert("penicillin", 1928);
        m.insert("vaccine", 1796);
        m.insert("x-ray", 1895);
        m.insert("anesthesia", 1846);
        m.insert("aspirin", 1899);
        m.insert("insulin", 1921);
        m.insert("dna", 1953);
        m.insert("gene", 1909);
        m.insert("vitamin", 1912);
        m.insert("calorie", 1824);
        m.insert("electron", 1897);
        m.insert("atom", 1803);
        m.insert("neutron", 1932);
        m.insert("radioactive", 1898);
        m.insert("plastic", 1907);
        m.insert("nylon", 1935);
        m.insert("dynamite", 1867);
        m.insert("biodiversity", 1985);
        // Medical instruments & drugs
        m.insert("stethoscope", 1816);
        m.insert("thermometer", 1714);
        m.insert("syringe", 1853);
        m.insert("morphine", 1804);
        m.insert("chloroform", 1831);
        m.insert("defibrillator", 1947);
        m.insert("ultrasound", 1956);
        m.insert("ibuprofen", 1961);
        m.insert("dialysis", 1943);
        m.insert("pacemaker", 1958);
        m.insert("prosthetic", 1912);
        m.insert("antiseptic", 1867);
        m.insert("bandaid", 1920);
        // Weapons
        m.insert("gatling", 1862);
        m.insert("torpedo", 1800);
        m.insert("landmine", 1277);
        m.insert("grenade", 1500);
        m.insert("napalm", 1942);
        m.insert("missile", 1611);
        m.insert("bayonet", 1670);
        m.insert("revolver", 1836);
        m.insert("carbine", 1605);
        m.insert("howitzer", 1695);
        m.insert("mortar", 1453);
        m.insert("musket", 1499);
        m.insert("flintlock", 1610);
        m.insert("matchlock", 1411);
        m.insert("blunderbuss", 1654);
        m.insert("shrapnel", 1784);
        // Clothing
        m.insert("jeans", 1873);
        m.insert("sneakers", 1895);
        m.insert("bra", 1914);
        m.insert("bikini", 1946);
        m.insert("hoodie", 1930);
        m.insert("tuxedo", 1888);
        m.insert("raincoat", 1830);
        m.insert("cardigan", 1868);
        m.insert("khaki", 1848);
        m.insert("parka", 1780);
        // Music
        m.insert("phonograph", 1877);
        m.insert("microphone", 1877);
        m.insert("amplifier", 1912);
        m.insert("synthesizer", 1955);
        m.insert("gramophone", 1887);
        m.insert("jukebox", 1927);
        m.insert("turntable", 1925);
        m.insert("headphones", 1910);
        m.insert("loudspeaker", 1898);
        m.insert("saxophone", 1846);
        m.insert("harmonica", 1821);
        m.insert("accordion", 1822);
        // Sports
        m.insert("basketball", 1891);
        m.insert("volleyball", 1895);
        m.insert("baseball", 1845);
        m.insert("tennis", 1874);
        m.insert("badminton", 1873);
        m.insert("hockey", 1875);
        m.insert("rugby", 1823);
        m.insert("cricket", 1598);
        // Food & drink
        m.insert("hamburger", 1884);
        m.insert("chocolate-chip", 1938);
        m.insert("sushi", 1820);
        m.insert("ketchup", 1812);
        m.insert("cornflakes", 1894);
        m.insert("pretzel", 1510);
        m.insert("croissant", 1838);
        m.insert("espresso", 1884);
        m.insert("cappuccino", 1930);
        m.insert("brunch", 1895);
        m.insert("cocktail", 1806);
        // Social & political
        m.insert("feminism", 1837);
        m.insert("socialism", 1832);
        m.insert("capitalism", 1850);
        m.insert("communism", 1840);
        m.insert("environmentalist", 1960);
        m.insert("genocide", 1944);
        m.insert("propaganda", 1622);
        m.insert("boycott", 1880);
        m.insert("sabotage", 1910);
        m.insert("fascism", 1919);
        m.insert("racism", 1902);
        m.insert("totalitarian", 1926);
        m.insert("anarchism", 1642);
        m.insert("imperialism", 1826);
        m.insert("nationalism", 1798);
        // Everyday items
        m.insert("zipper", 1917);
        m.insert("refrigerator", 1913);
        m.insert("microwave", 1947);
        m.insert("dishwasher", 1886);
        m.insert("toaster", 1909);
        m.insert("blender", 1922);
        m.insert("vacuum", 1901);
        m.insert("lightbulb", 1879);
        m.insert("battery", 1800);
        m.insert("photograph", 1839);
        m.insert("camera", 1816);
        m.insert("wristwatch", 1868);
        m.insert("thermos", 1904);
        m.insert("ballpoint", 1938);
        m.insert("stapler", 1877);
        m.insert("paperclip", 1867);
        m.insert("eraser", 1770);
        // Textiles & fashion
        m.insert("polyester", 1941);
        m.insert("spandex", 1958);
        m.insert("velcro", 1941);
        m.insert("rayon", 1924);
        m.insert("lycra", 1958);
        // Food & drink (additional)
        m.insert("pasteurize", 1862);
        m.insert("margarine", 1869);
        m.insert("coca-cola", 1886);
        // Language / slang
        m.insert("groovy", 1937);
        m.insert("dude", 1883);
        // Misc
        m.insert("automation", 1946);
        m.insert("feedback", 1920);
        m.insert("logistics", 1846);
        m.insert("ok", 1839);
        m.insert("pipeline", 1900);
        m.insert("deadline", 1864);
        m.insert("freelance", 1820);
        m.insert("bureaucracy", 1818);
        m.insert("stereotype", 1798);
        m.insert("silhouette", 1783);
        m.insert("panorama", 1791);
        m.insert("electricity", 1646);
        m.insert("barometer", 1666);
        m.insert("pendulum", 1643);
        m.insert("telescope", 1611);
        m.insert("microscope", 1625);
        m.insert("encyclopedia", 1541);
        m
    })
}

fn multi_word_milestones() -> &'static Vec<(&'static str, i32)> {
    static M: OnceLock<Vec<(&str, i32)>> = OnceLock::new();
    M.get_or_init(|| {
        vec![
            ("cell phone", 1983),
            ("social media", 2004),
            ("world wide web", 1991),
            ("virtual reality", 1987),
            ("artificial intelligence", 1956),
            ("credit card", 1950),
            ("fast food", 1951),
            ("global warming", 1975),
            ("assembly line", 1913),
            ("birth control", 1914),
            ("machine learning", 1959),
            ("nuclear power", 1945),
            ("space shuttle", 1981),
            ("t-shirt", 1904),
            ("hot-dog", 1867),
            ("mp3 player", 1998),
            ("blood transfusion", 1818),
            ("contact lens", 1887),
            ("dry cleaning", 1855),
            ("ice cream", 1744),
            ("chewing gum", 1848),
            ("barbed wire", 1873),
            ("sewing machine", 1846),
            ("typewriter", 1868),
            ("washing machine", 1851),
            ("printing press", 1440),
            ("steam engine", 1712),
            ("internal combustion", 1860),
            ("stock market", 1611),
            ("central bank", 1668),
            ("paper money", 1690),
        ]
    })
}

/// Concepts that are anachronistic in medieval settings (target_year < 1500),
/// even if the specific word existed earlier in other contexts.
fn medieval_anachronisms() -> &'static Vec<(&'static str, &'static str)> {
    static M: OnceLock<Vec<(&str, &str)>> = OnceLock::new();
    M.get_or_init(|| {
        vec![
            ("democracy", "modern democratic concept"),
            ("evolution", "evolutionary theory"),
            ("psychology", "psychological science"),
            ("genetics", "genetic science"),
            ("bureaucracy", "bureaucratic systems"),
            ("constitution", "constitutional government"),
            ("parliament", "parliamentary government"),
            ("republic", "republican government"),
            ("capitalism", "capitalist economy"),
            ("socialism", "socialist ideology"),
            ("communism", "communist ideology"),
            ("nationalism", "nationalist movement"),
            ("revolution", "political revolution concept"),
            ("ideology", "ideological framework"),
            ("civilization", "civilizational concept"),
            ("education", "formal education system"),
            ("university", "university institution"),
            ("newspaper", "printed newspaper"),
            ("patent", "patent system"),
            ("copyright", "copyright law"),
            ("insurance", "insurance system"),
            ("factory", "industrial factory"),
            ("corporation", "corporate entity"),
            ("investment", "financial investment"),
            ("inflation", "economic inflation"),
        ]
    })
}

/// Slang terms that are anachronistic if used before their era.
fn slang_milestones() -> &'static Vec<(&'static str, i32)> {
    static M: OnceLock<Vec<(&str, i32)>> = OnceLock::new();
    M.get_or_init(|| {
        vec![
            ("cool", 1933),
            ("okay", 1839),
            ("awesome", 1980),
            ("bummer", 1966),
            ("chill", 1979),
            ("vibe", 1967),
            ("gig", 1926),
            ("hip", 1904),
            ("jazz", 1912),
            ("punk", 1596),
            ("geek", 1916),
            ("nerd", 1951),
            ("binge", 1854),
            ("hype", 1931),
            ("mob", 1688),
            ("radar", 1935),
            ("snafu", 1941),
        ]
    })
}

fn is_anachronistic(introduced_year: i32, target_year: i32) -> bool {
    introduced_year > target_year + GRACE_PERIOD_YEARS
}

/// Analyze a scene's tokens for anachronistic words given a target historical year.
/// Returns empty if target_year is 0 (no target set).
pub fn analyze_scene(tokens: &[String], target_year: i32, scene_id: usize) -> Vec<AnachronismItem> {
    if target_year == 0 {
        return Vec::new();
    }

    let mut results = Vec::new();
    let ms = milestones();

    // Build lowercase full text for multi-word matching
    let full_text: String = tokens
        .iter()
        .map(|t| t.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");

    // Check multi-word terms first
    for &(term, introduced_year) in multi_word_milestones() {
        if full_text.contains(term) && is_anachronistic(introduced_year, target_year) {
            results.push(AnachronismItem {
                word: term.to_string(),
                introduced_year,
                target_year,
                scene_id,
            });
        }
    }

    // Check single-word terms
    for token in tokens {
        let lower = token.to_lowercase();
        if let Some(&introduced_year) = ms.get(lower.as_str())
            && is_anachronistic(introduced_year, target_year) {
                results.push(AnachronismItem {
                    word: token.clone(),
                    introduced_year,
                    target_year,
                    scene_id,
                });
            }
    }

    // Slang terms (context-dependent, only flag if clearly used as slang)
    for &(term, introduced_year) in slang_milestones() {
        if is_anachronistic(introduced_year, target_year) {
            for token in tokens {
                if token.to_lowercase() == term {
                    results.push(AnachronismItem {
                        word: token.clone(),
                        introduced_year,
                        target_year,
                        scene_id,
                    });
                    break;
                }
            }
        }
    }

    // Medieval period-specific vocabulary warnings
    if target_year < 1500 {
        for &(concept, description) in medieval_anachronisms() {
            // Skip terms already covered in milestones
            if ms.contains_key(concept) {
                continue;
            }
            for token in tokens {
                if token.to_lowercase() == concept {
                    results.push(AnachronismItem {
                        word: format!("{} ({})", token, description),
                        introduced_year: 1500,
                        target_year,
                        scene_id,
                    });
                    break;
                }
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modern_word_in_medieval() {
        let tokens: Vec<String> = vec!["The", "knight", "checked", "his", "smartphone"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1200, 0);
        assert!(results.iter().any(|a| a.word == "smartphone"));
        assert_eq!(
            results
                .iter()
                .find(|a| a.word == "smartphone")
                .unwrap()
                .introduced_year,
            2007
        );
    }

    #[test]
    fn test_word_within_grace_period() {
        // "telephone" introduced 1876; target 1860 + 20 grace = 1880, so not anachronistic
        let tokens: Vec<String> = vec!["telephone"].into_iter().map(String::from).collect();
        let results = analyze_scene(&tokens, 1860, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_word_before_grace_period() {
        // "telephone" introduced 1876; target 1800 + 20 grace = 1820, so anachronistic
        let tokens: Vec<String> = vec!["telephone"].into_iter().map(String::from).collect();
        let results = analyze_scene(&tokens, 1800, 0);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_multi_word_term() {
        let tokens: Vec<String> = vec!["He", "used", "social", "media", "daily"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1900, 0);
        assert!(results.iter().any(|a| a.word == "social media"));
    }

    #[test]
    fn test_no_target_year() {
        let tokens: Vec<String> = vec!["smartphone"].into_iter().map(String::from).collect();
        let results = analyze_scene(&tokens, 0, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_case_insensitive() {
        let tokens: Vec<String> = vec!["INTERNET"].into_iter().map(String::from).collect();
        let results = analyze_scene(&tokens, 1500, 0);
        assert!(results.iter().any(|a| a.word == "INTERNET"));
    }

    #[test]
    fn test_no_false_positives_for_era() {
        // "atom" introduced 1803; target 1900 should be fine
        let tokens: Vec<String> = vec!["atom", "electron", "vaccine"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1900, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_medical_instruments() {
        let tokens: Vec<String> = vec!["stethoscope", "defibrillator"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1700, 0);
        assert!(results.iter().any(|a| a.word == "stethoscope"));
        assert!(results.iter().any(|a| a.word == "defibrillator"));
    }

    #[test]
    fn test_weapons() {
        let tokens: Vec<String> = vec!["gatling", "bayonet", "napalm"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1600, 0);
        assert!(results.iter().any(|a| a.word == "gatling"));
        assert!(results.iter().any(|a| a.word == "napalm"));
        // bayonet introduced 1670, target 1600 + 20 grace = 1620, so anachronistic
        assert!(results.iter().any(|a| a.word == "bayonet"));
    }

    #[test]
    fn test_clothing() {
        let tokens: Vec<String> = vec!["jeans", "bikini"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1800, 0);
        assert!(results.iter().any(|a| a.word == "jeans"));
        assert!(results.iter().any(|a| a.word == "bikini"));
    }

    #[test]
    fn test_music_terms() {
        let tokens: Vec<String> = vec!["phonograph", "synthesizer"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1800, 0);
        assert!(results.iter().any(|a| a.word == "phonograph"));
        assert!(results.iter().any(|a| a.word == "synthesizer"));
    }

    #[test]
    fn test_sports() {
        let tokens: Vec<String> = vec!["basketball", "volleyball"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1800, 0);
        assert!(results.iter().any(|a| a.word == "basketball"));
        assert!(results.iter().any(|a| a.word == "volleyball"));
    }

    #[test]
    fn test_food() {
        let tokens: Vec<String> = vec!["hamburger", "espresso"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1800, 0);
        assert!(results.iter().any(|a| a.word == "hamburger"));
        assert!(results.iter().any(|a| a.word == "espresso"));
    }

    #[test]
    fn test_slang_cool_in_medieval() {
        let tokens: Vec<String> = vec!["That", "is", "cool"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1200, 0);
        assert!(results.iter().any(|a| a.word == "cool"));
    }

    #[test]
    fn test_slang_okay_before_1839() {
        let tokens: Vec<String> = vec!["okay"].into_iter().map(String::from).collect();
        let results = analyze_scene(&tokens, 1700, 0);
        assert!(results.iter().any(|a| a.word == "okay"));
    }

    #[test]
    fn test_slang_awesome_modern() {
        let tokens: Vec<String> = vec!["awesome"].into_iter().map(String::from).collect();
        let results = analyze_scene(&tokens, 1900, 0);
        assert!(results.iter().any(|a| a.word == "awesome"));
    }

    #[test]
    fn test_medieval_concept_warnings() {
        let tokens: Vec<String> = vec!["The", "kingdom", "had", "a", "democracy"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1200, 0);
        assert!(results
            .iter()
            .any(|a| a.word.contains("democracy") && a.word.contains("modern democratic")));
    }

    #[test]
    fn test_medieval_psychology_warning() {
        let tokens: Vec<String> = vec!["psychology"].into_iter().map(String::from).collect();
        let results = analyze_scene(&tokens, 1100, 0);
        assert!(results.iter().any(|a| a.word.contains("psychology")));
    }

    #[test]
    fn test_medieval_concepts_not_flagged_in_modern() {
        // Medieval-only concepts should not be flagged for modern settings
        let tokens: Vec<String> = vec!["democracy", "evolution", "psychology"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1900, 0);
        // These words are not in the regular milestones, so should not appear
        assert!(!results.iter().any(|a| a.word.contains("democracy")));
    }

    #[test]
    fn test_new_multi_word_terms() {
        let tokens: Vec<String> = vec!["She", "used", "a", "sewing", "machine"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1700, 0);
        assert!(results.iter().any(|a| a.word == "sewing machine"));
    }

    #[test]
    fn test_electronics_terms() {
        let tokens: Vec<String> = vec!["semiconductor", "microprocessor"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = analyze_scene(&tokens, 1900, 0);
        assert!(results.iter().any(|a| a.word == "semiconductor"));
        assert!(results.iter().any(|a| a.word == "microprocessor"));
    }

    #[test]
    fn test_total_milestone_count() {
        let single = milestones().len();
        let multi = multi_word_milestones().len();
        let slang = slang_milestones().len();
        let medieval = medieval_anachronisms().len();
        // Verify we have substantially more entries than before
        assert!(
            single >= 180,
            "Expected 180+ single-word entries, got {}",
            single
        );
        assert!(
            multi >= 25,
            "Expected 25+ multi-word entries, got {}",
            multi
        );
        assert!(slang >= 10, "Expected 10+ slang entries, got {}", slang);
        assert!(
            medieval >= 20,
            "Expected 20+ medieval entries, got {}",
            medieval
        );
    }
}
