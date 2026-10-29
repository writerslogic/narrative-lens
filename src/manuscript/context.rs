use super::*;
use crate::{
    Error, Result,
    craft::{grammar, lexicon::WordMatcher, readability, syntax_tension},
    onnx::{
        self, InferenceBackend,
        checked::{self, FailureKind, ModelFailure, ModelKind},
        model::ModelIdentity,
    },
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct GrammarDependencies {
    selection: grammar::Selection,
    limit: usize,
}
#[derive(Debug, Clone, PartialEq)]
struct WorldDependencies {
    characters: Vec<String>,
    corrections: Vec<crate::continuity::entity::EntityCorrection>,
    limit: usize,
    selection: continuity::EventSelection,
}
#[derive(Debug, Clone)]
struct GrammarData {
    dependencies: Arc<GrammarDependencies>,
    intentions: u8,
    findings: Vec<grammar::GrammarFinding>,
    limited: bool,
}
#[derive(Debug, Clone)]
struct WorldData {
    dependencies: Arc<WorldDependencies>,
    events: Vec<continuity::Event>,
    limited: bool,
}
#[derive(Debug, Clone)]
struct Intermediate {
    grammar: Arc<GrammarData>,
    world: Arc<WorldData>,
    metrics: SceneMetrics,
    sentences: Arc<Vec<(usize, usize)>>,
}
impl Intermediate {
    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.metrics.scene_id.capacity()
            + self.sentences.capacity() * std::mem::size_of::<(usize, usize)>()
            + std::mem::size_of::<GrammarData>()
            + self.grammar.findings.capacity() * std::mem::size_of::<grammar::GrammarFinding>()
            + self
                .grammar
                .findings
                .iter()
                .map(|g| {
                    g.message.capacity()
                        + g.replacements.capacity() * std::mem::size_of::<String>()
                        + g.replacements.iter().map(String::capacity).sum::<usize>()
                })
                .sum::<usize>()
            + std::mem::size_of::<WorldData>()
            + self.world.events.capacity() * std::mem::size_of::<continuity::Event>()
            + self
                .world
                .events
                .iter()
                .map(|e| {
                    e.character.capacity()
                        + match &e.kind {
                            continuity::EventKind::Location(s)
                            | continuity::EventKind::Attribute(s)
                            | continuity::EventKind::Knowledge(s)
                            | continuity::EventKind::ObjectTransfer(s) => s.capacity(),
                            continuity::EventKind::Object(a, b) => a.capacity() + b.capacity(),
                            _ => 0,
                        }
                })
                .sum::<usize>()
            + 6 * std::mem::size_of::<usize>()
    }
}
#[derive(Debug)]
struct Entry {
    text: String,
    data: Intermediate,
    charge: usize,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CacheStats {
    pub reused_scenes: usize,
    pub recomputed_scenes: usize,
    pub entries: usize,
    pub retained_bytes: usize,
    pub budget_bytes: usize,
    pub recomputed_metrics_scenes: usize,
    pub recomputed_grammar_scenes: usize,
    pub recomputed_world_scenes: usize,
}
#[derive(Debug)]
pub struct AnalysisContext {
    entries: BTreeMap<String, Entry>,
    grammar: Option<Arc<GrammarDependencies>>,
    world: Option<Arc<WorldDependencies>>,
    budget: usize,
    stats: CacheStats,
}
impl Default for AnalysisContext {
    fn default() -> Self {
        Self::new(16 * 1024 * 1024).expect("valid default cache budget")
    }
}
impl AnalysisContext {
    pub fn new(max_cache_bytes: usize) -> Result<Self> {
        if max_cache_bytes > 256 * 1024 * 1024 {
            return Err(Error::InvalidInput("cache budget exceeds 256 MiB".into()));
        }
        Ok(Self {
            entries: BTreeMap::new(),
            grammar: None,
            world: None,
            budget: max_cache_bytes,
            stats: CacheStats {
                budget_bytes: max_cache_bytes,
                ..Default::default()
            },
        })
    }
    pub fn clear(&mut self) {
        self.entries.clear();
        self.grammar = None;
        self.world = None;
        self.stats = CacheStats {
            budget_bytes: self.budget,
            ..Default::default()
        };
    }
    pub fn stats(&self) -> CacheStats {
        self.stats.clone()
    }
    pub fn analyze(&mut self, m: &Manuscript) -> Result<Report> {
        self.analyze_with_backend(m, onnx::backend())
    }
    pub fn analyze_with_backend(
        &mut self,
        m: &Manuscript,
        backend: &dyn InferenceBackend,
    ) -> Result<Report> {
        m.validate()?;
        let c = &m.configuration;
        let gram = GrammarDependencies {
            selection: grammar::Selection {
                repetition: c.selected("NL-GRAMMAR-REPETITION")
                    && Severity::Warning >= c.minimum_severity,
                spelling: c.selected("NL-GRAMMAR-SPELLING")
                    && Severity::Suggestion >= c.minimum_severity,
                whitespace: c.selected("NL-GRAMMAR-WHITESPACE")
                    && Severity::Info >= c.minimum_severity,
                dictionary: Arc::new(c.dictionary.iter().map(|s| s.to_lowercase()).collect()),
            },
            limit: c.limits.scene_candidates,
        };
        let world = WorldDependencies {
            characters: m.known_characters.clone(),
            corrections: c.entity_corrections.clone(),
            limit: c.limits.scene_candidates.saturating_mul(2),
            selection: continuity::EventSelection {
                location: c.selected("NL-CONTINUITY-LOCKED")
                    && Severity::Warning >= c.minimum_severity,
                attribute: c.selected("NL-CONTINUITY-ATTRIBUTE")
                    && Severity::Warning >= c.minimum_severity,
                object: c.selected("NL-CONTINUITY-OBJECT")
                    && Severity::Warning >= c.minimum_severity,
                knowledge: c.chronology,
            },
        };
        let grammar = self
            .grammar
            .as_ref()
            .filter(|g| ***g == gram)
            .cloned()
            .unwrap_or_else(|| Arc::new(gram));
        let world = self
            .world
            .as_ref()
            .filter(|w| ***w == world)
            .cloned()
            .unwrap_or_else(|| Arc::new(world));
        let metadata = grammar
            .selection
            .dictionary
            .iter()
            .map(|s| s.capacity() + std::mem::size_of::<String>() + 64)
            .sum::<usize>()
            + world.characters.iter().map(String::capacity).sum::<usize>()
            + world.characters.capacity() * std::mem::size_of::<String>()
            + world
                .corrections
                .iter()
                .map(|correction| {
                    correction.surface.capacity()
                        + match &correction.action {
                            crate::continuity::entity::CorrectionAction::Assign { canonical } => {
                                canonical.capacity()
                            }
                            _ => 0,
                        }
                })
                .sum::<usize>()
            + world.corrections.capacity()
                * std::mem::size_of::<crate::continuity::entity::EntityCorrection>()
            + std::mem::size_of::<GrammarDependencies>()
            + std::mem::size_of::<WorldDependencies>()
            + 64;
        let ids: BTreeSet<_> = m.scenes.iter().map(|s| &s.id).collect();
        self.entries.retain(|id, _| ids.contains(id));
        self.stats = CacheStats {
            budget_bytes: self.budget,
            ..Default::default()
        };
        let mut parser = None;
        let mut data = Vec::with_capacity(m.scenes.len());
        for scene in &m.scenes {
            let old = self
                .entries
                .get(&scene.id)
                .filter(|e| e.text == scene.text)
                .map(|e| e.data.clone());
            let intentions = u8::from(scene.intentions.contains(&Intention::Repetition))
                | (u8::from(scene.intentions.contains(&Intention::Dialect)) * 2);
            let grammar_hit = old.as_ref().is_some_and(|d| {
                Arc::ptr_eq(&d.grammar.dependencies, &grammar) && d.grammar.intentions == intentions
            });
            let world_hit = old
                .as_ref()
                .is_some_and(|d| Arc::ptr_eq(&d.world.dependencies, &world));
            let grammar_data = if grammar_hit {
                Arc::clone(&old.as_ref().expect("cached grammar").grammar)
            } else {
                let mut selection = grammar.selection.clone();
                selection.repetition &= intentions & 1 == 0;
                selection.spelling &= intentions & 2 == 0;
                let (findings, limited) =
                    grammar::check_selected(&scene.text, grammar.limit, &selection);
                self.stats.recomputed_grammar_scenes += 1;
                Arc::new(GrammarData {
                    dependencies: Arc::clone(&grammar),
                    intentions,
                    findings,
                    limited,
                })
            };
            let (metrics, sentences) = if let Some(d) = &old {
                (d.metrics.clone(), Arc::clone(&d.sentences))
            } else {
                self.stats.recomputed_metrics_scenes += 1;
                let r = readability::compute_readability(&scene.text);
                let p = readability::compute_pacing_metrics(&scene.text, 0);
                (
                    SceneMetrics {
                        scene_id: scene.id.clone(),
                        word_count: r.word_count,
                        sentence_count: r.sentence_count,
                        grade: r.fkgl,
                        sentence_words: p.velocity,
                        dialogue_ratio: p.dialogue_ratio,
                        tension: syntax_tension::analyze_scene_text(&scene.text).score,
                    },
                    Arc::new(crate::substrate::utils::sentence_spans(&scene.text)),
                )
            };
            let world_data = if world_hit {
                Arc::clone(&old.as_ref().expect("cached world").world)
            } else {
                let mut events = Vec::new();
                if world.selection.enabled() {
                    if parser.is_none() {
                        parser = Some(continuity::Parser::new(
                            &world.characters,
                            &world.corrections,
                        )?);
                    }
                    events = parser.as_ref().expect("world parser").extract_bounded(
                        &scene.text,
                        &sentences,
                        world.limit,
                        world.selection,
                    );
                }
                let limited = events.len() > world.limit;
                events.truncate(world.limit);
                self.stats.recomputed_world_scenes += 1;
                Arc::new(WorldData {
                    dependencies: Arc::clone(&world),
                    events,
                    limited,
                })
            };
            if old.is_some() && grammar_hit && world_hit {
                self.stats.reused_scenes += 1;
            } else {
                self.stats.recomputed_scenes += 1;
            }
            let intermediate = Intermediate {
                grammar: grammar_data,
                world: world_data,
                metrics,
                sentences,
            };
            let charge = intermediate.retained_bytes()
                + scene.text.len()
                + scene.id.len()
                + std::mem::size_of::<Entry>()
                + 256;
            if metadata.saturating_add(charge) <= self.budget {
                self.entries.insert(
                    scene.id.clone(),
                    Entry {
                        text: scene.text.clone(),
                        data: intermediate.clone(),
                        charge,
                    },
                );
            } else {
                self.entries.remove(&scene.id);
            }
            data.push(intermediate);
        }
        self.stats.retained_bytes =
            metadata + self.entries.values().map(|e| e.charge).sum::<usize>();
        while self.stats.retained_bytes > self.budget && !self.entries.is_empty() {
            let (_, entry) = self.entries.pop_first().expect("nonempty cache");
            self.stats.retained_bytes -= entry.charge;
        }
        if self.entries.is_empty() {
            self.grammar = None;
            self.world = None;
            self.stats.retained_bytes = 0;
        } else {
            self.grammar = Some(grammar);
            self.world = Some(world);
        }
        self.stats.entries = self.entries.len();
        build_report(m, &data, backend)
    }
}

fn finding(
    document_id: &str,
    rule: &str,
    analyzer: &str,
    severity: Severity,
    message: String,
    suggestion: &str,
    confidence: f64,
    scope: Scope,
    evidence: Vec<Evidence>,
) -> Finding {
    Finding {
        document_id: document_id.into(),
        rule_id: rule.into(),
        analyzer: analyzer.into(),
        severity,
        message,
        suggestion: suggestion.into(),
        confidence: Confidence::heuristic(
            confidence,
            "manuscript-v1; uncalibrated engineering score",
        ),
        scope,
        evidence,
        fingerprint: String::new(),
    }
}
fn sample_evidence(m: &Manuscript, scene: &Scene, d: &Intermediate) -> Vec<Evidence> {
    d.sentences
        .iter()
        .max_by_key(|(start, end)| end - start)
        .map(|&(start, end)| {
            let end = scene.text[start..end]
                .char_indices()
                .nth(512)
                .map_or(end, |(offset, _)| start + offset);
            Evidence::new(
                &m.document_id,
                &scene.id,
                &scene.text,
                start,
                end,
                EvidenceRole::Context,
            )
            .expect("original sentence sample")
        })
        .into_iter()
        .collect()
}
fn build_report(
    m: &Manuscript,
    data: &[Intermediate],
    backend: &dyn InferenceBackend,
) -> Result<Report> {
    let config = &m.configuration;
    let mut collector = super::collector::Collector::new(m);
    let mut diagnostics = Vec::new();
    let mut coverage = Vec::new();
    let terms = config
        .lexicon
        .iter()
        .map(|s| s.to_ascii_lowercase())
        .collect::<BTreeSet<_>>();
    let lexicon = WordMatcher::new(&terms.iter().map(String::as_str).collect::<Vec<_>>());
    let grammar_selected = [
        "NL-GRAMMAR-REPETITION",
        "NL-GRAMMAR-SPELLING",
        "NL-GRAMMAR-WHITESPACE",
    ]
    .iter()
    .any(|r| {
        config.selected(r)
            && match *r {
                "NL-GRAMMAR-REPETITION" => Severity::Warning,
                "NL-GRAMMAR-SPELLING" => Severity::Suggestion,
                _ => Severity::Info,
            } >= config.minimum_severity
    });
    let nonempty = m
        .scenes
        .iter()
        .filter(|s| !s.text.trim().is_empty())
        .count();
    let mut sufficient = 0;
    for (scene, d) in m.scenes.iter().zip(data) {
        if grammar_selected {
            if d.grammar.limited {
                diagnostics.push(Diagnostic {
                    code: "grammar_candidate_limit".into(),
                    message: format!(
                        "Scene grammar candidates exceed {}.",
                        config.limits.scene_candidates
                    ),
                    scene_ids: vec![scene.id.clone()],
                });
            }
            if !d.grammar.findings.is_empty() {
                let offsets = crate::substrate::offsets::OffsetIndex::new(&scene.text)?;
                for g in &d.grammar.findings {
                    collector.emit(finding::adapt_grammar_indexed(
                        &m.document_id,
                        &scene.id,
                        &offsets,
                        g,
                    )?);
                }
            }
        }
        if d.metrics.word_count >= config.min_scene_words() {
            sufficient += 1;
            if config.selected("NL-READABILITY-COMPLEX")
                && d.metrics.grade > config.max_grade()
                && !scene.intentions.contains(&Intention::ComplexProse)
            {
                collector.emit(finding(
                    &m.document_id,
                    "NL-READABILITY-COMPLEX",
                    "readability",
                    Severity::Suggestion,
                    format!(
                        "Scene grade {:.1} exceeds the configured maximum {:.1}.",
                        d.metrics.grade,
                        config.max_grade()
                    ),
                    "Review sentence length and vocabulary against the intended audience.",
                    0.8,
                    Scope::Scene {
                        scene_id: scene.id.clone(),
                    },
                    sample_evidence(m, scene, d),
                ));
            }
            if config.selected("NL-PACING-LONG")
                && d.metrics.sentence_words > config.max_sentence_words()
                && !scene.intentions.contains(&Intention::LongSentences)
            {
                collector.emit(finding(
                    &m.document_id,
                    "NL-PACING-LONG",
                    "pacing",
                    Severity::Suggestion,
                    format!(
                        "Scene averages {:.1} words per sentence, above {:.1}.",
                        d.metrics.sentence_words,
                        config.max_sentence_words()
                    ),
                    "Review sentence boundaries and the intended pace.",
                    0.8,
                    Scope::Scene {
                        scene_id: scene.id.clone(),
                    },
                    sample_evidence(m, scene, d),
                ));
            }
        }
        if config.selected("NL-PROSE-LEXICON") && !config.lexicon.is_empty() {
            for (index, word) in lexicon.find_iter(&scene.text).enumerate() {
                if index == config.limits.scene_candidates {
                    diagnostics.push(Diagnostic {
                        code: "lexicon_candidate_limit".into(),
                        message: format!(
                            "Scene editorial-term candidates exceed {}.",
                            config.limits.scene_candidates
                        ),
                        scene_ids: vec![scene.id.clone()],
                    });
                    break;
                }
                collector.emit(finding(
                    &m.document_id,
                    "NL-PROSE-LEXICON",
                    "prose",
                    Severity::Suggestion,
                    "Author-configured editorial term.".into(),
                    "Review this term against your lexicon.",
                    0.8,
                    Scope::Passage {
                        scene_id: scene.id.clone(),
                    },
                    vec![Evidence::new(
                        &m.document_id,
                        &scene.id,
                        &scene.text,
                        word.start,
                        word.end,
                        EvidenceRole::Primary,
                    )?],
                ));
            }
        }
    }
    for (analyzer, selected, analyzed) in [
        ("grammar", grammar_selected, nonempty),
        (
            "readability",
            config.selected("NL-READABILITY-COMPLEX"),
            sufficient,
        ),
        ("pacing", config.selected("NL-PACING-LONG"), sufficient),
        (
            "prose",
            config.selected("NL-PROSE-LEXICON") && !config.lexicon.is_empty(),
            nonempty,
        ),
    ] {
        coverage.push(Coverage {
            diagnostics: if !selected {
                vec!["rule_not_selected_or_lexicon_empty".into()]
            } else if analyzed < nonempty {
                vec!["below_minimum_scene_words".into()]
            } else {
                Vec::new()
            },
            ..Coverage::scenes(analyzer, selected, analyzed, nonempty, m.scenes.len())
        });
    }
    let structure_selected = config.selected("NL-STRUCTURE-MISSING-BEAT");
    let structure_available = nonempty >= config.thresholds.min_structure_scenes.unwrap_or(3);
    let nonlinear = nonempty > 0
        && m.scenes
            .iter()
            .filter(|s| !s.text.trim().is_empty())
            .all(|s| s.intentions.contains(&Intention::Nonlinear));
    let open_ending = m
        .scenes
        .iter()
        .rfind(|s| !s.text.trim().is_empty())
        .is_some_and(|s| s.intentions.contains(&Intention::OpenEnding));
    if structure_selected && structure_available && !nonlinear {
        let report = crate::structure::structure::validate_structure(
            &config.structure_template,
            data.iter().map(|d| d.metrics.tension).collect(),
            m.scenes.iter().map(|s| s.purpose.clone()).collect(),
        )?;
        for beat in report.missing_beats {
            if open_ending
                && ["resolution", "return", "final image", "denouement"]
                    .iter()
                    .any(|word| beat.to_lowercase().contains(word))
            {
                continue;
            }
            collector.emit(finding(
                &m.document_id,
                "NL-STRUCTURE-MISSING-BEAT",
                "structure",
                Severity::Suggestion,
                format!(
                    "Template {} has no supported match for {beat}.",
                    config.structure_template
                ),
                "Supply scene purposes or review whether this template suits the manuscript.",
                0.8,
                Scope::Manuscript {
                    scene_ids: m.scenes.iter().map(|s| s.id.clone()).collect(),
                },
                Vec::new(),
            ));
        }
    }
    coverage.push(Coverage {
        diagnostics: if !structure_selected {
            vec!["rule_not_selected".into()]
        } else if nonlinear {
            vec!["author_intends_nonlinear_structure".into()]
        } else if !structure_available {
            vec!["below_minimum_structure_scenes".into()]
        } else if open_ending {
            vec!["author_intends_open_ending".into()]
        } else {
            Vec::new()
        },
        ..Coverage::scenes(
            "structure",
            structure_selected && !nonlinear,
            if structure_available { nonempty } else { 0 },
            nonempty,
            m.scenes.len(),
        )
    });
    let physical = [
        "NL-CONTINUITY-LOCKED",
        "NL-CONTINUITY-ATTRIBUTE",
        "NL-CONTINUITY-OBJECT",
    ]
    .iter()
    .any(|r| config.selected(r));
    let events = data
        .iter()
        .map(|d| d.world.events.as_slice())
        .collect::<Vec<_>>();
    let limits = data.iter().map(|d| d.world.limited).collect::<Vec<_>>();
    for (scene, limited) in m.scenes.iter().zip(&limits) {
        if *limited {
            diagnostics.push(Diagnostic {
                code: "world_candidate_limit".into(),
                message: format!(
                    "Scene world events exceed {}.",
                    config.limits.scene_candidates * 2
                ),
                scene_ids: vec![scene.id.clone()],
            });
        }
    }
    let (chronology, mut world_coverage) = continuity::analyze(
        m,
        &events,
        &limits,
        &mut |f| collector.emit(f),
        &mut diagnostics,
    );
    if !physical {
        world_coverage.status = Status::Unavailable;
        world_coverage.analyzed_units = 0;
        world_coverage.diagnostics.push("rules_not_selected".into());
    }
    coverage.push(world_coverage);
    let model_identities =
        analyze_nli(m, data, backend, &mut |f| collector.emit(f), &mut coverage)?;
    let (findings, suppressed, summary) = collector.finish();
    if summary.omitted_findings > 0 || summary.omitted_suppressed > 0 {
        diagnostics.push(Diagnostic{code:"finding_limit".into(),message:format!("{} active and {} suppressed findings omitted; total severity counts remain available.",summary.omitted_findings,summary.omitted_suppressed),scene_ids:Vec::new()});
    }
    for c in &mut coverage {
        let partial = diagnostics.iter().any(|d| match d.code.as_str() {
            "finding_limit" => true,
            "grammar_candidate_limit" => c.analyzer == "grammar",
            "lexicon_candidate_limit" => c.analyzer == "prose",
            "world_candidate_limit" | "chronology_ambiguous" | "ambiguous_constraint" => {
                c.analyzer == "continuity"
            }
            _ => false,
        });
        if partial && c.status == Status::Complete {
            c.status = Status::HeuristicFallback;
        }
    }
    let report = Report {
        schema_version: "2".into(),
        configuration: Some(config.clone()),
        model_identities,
        document_id: m.document_id.clone(),
        scene_ids: m.scenes.iter().map(|s| s.id.clone()).collect(),
        summary,
        findings,
        suppressed,
        metrics: data.iter().map(|d| d.metrics.clone()).collect(),
        coverage,
        diagnostics,
        chronology,
    };
    report.validate()?;
    Ok(report)
}

fn analyze_nli(
    m: &Manuscript,
    data: &[Intermediate],
    backend: &dyn InferenceBackend,
    emit: &mut dyn FnMut(Finding),
    coverage: &mut Vec<Coverage>,
) -> Result<Vec<ModelIdentity>> {
    let config = &m.configuration;
    let disabled = config.nli == ModelPolicy::Disabled || !config.selected("NL-CONTINUITY-NLI");
    if disabled {
        coverage.push(Coverage {
            analyzer: "nli".into(),
            status: Status::Unavailable,
            total_units: 0,
            analyzed_units: 0,
            unit: "candidate_pairs".into(),
            diagnostics: vec!["model_analysis_disabled_or_rule_not_selected".into()],
        });
        return Ok(Vec::new());
    }
    let mut model_identities = Vec::new();
    let limits = &config.limits;
    let mut oversized = 0usize;
    let mut pairs = Vec::new();
    let mut total_pairs = 0usize;
    // A bounded adjacent-scene candidate window. Model findings are deliberately
    // unavailable under chronology: semantic pairs have no reliable lane state.
    if !config.chronology {
        for i in 1..data.len() {
            for previous in i.saturating_sub(limits.nli_window_scenes)..i {
                if !m.scenes[previous]
                    .intentions
                    .contains(&Intention::Supernatural)
                    && !m.scenes[i].intentions.contains(&Intention::Supernatural)
                {
                    total_pairs = total_pairs.saturating_add(
                        data[previous]
                            .sentences
                            .len()
                            .saturating_mul(data[i].sentences.len()),
                    );
                }
                for &(a, b) in data[previous]
                    .sentences
                    .iter()
                    .take(limits.nli_sentences_per_scene)
                {
                    for &(c, d) in data[i]
                        .sentences
                        .iter()
                        .take(limits.nli_sentences_per_scene)
                    {
                        if pairs.len() == limits.nli_pairs {
                            break;
                        }
                        if !m.scenes[previous]
                            .intentions
                            .contains(&Intention::Supernatural)
                            && !m.scenes[i].intentions.contains(&Intention::Supernatural)
                        {
                            if b - a > limits.model_input_bytes || d - c > limits.model_input_bytes
                            {
                                oversized += 1;
                                continue;
                            }
                            pairs.push((previous, a, b, i, c, d));
                        }
                    }
                }
            }
        }
    }
    let mut cov = Coverage {
        analyzer: "nli".into(),
        status: Status::Unavailable,
        total_units: total_pairs,
        analyzed_units: 0,
        unit: "candidate_pairs".into(),
        diagnostics: Vec::new(),
    };
    if disabled {
        cov.diagnostics
            .push("model_analysis_disabled_or_rule_not_selected".into());
    } else if pairs.is_empty() {
        cov.status = Status::InsufficientInput;
        cov.diagnostics.push(
            if config.chronology {
                "nli_not_supported_with_chronology"
            } else {
                "no_adjacent_sentence_pairs"
            }
            .into(),
        );
    } else {
        let input = pairs
            .iter()
            .map(|(i, a, b, j, c, d)| (&m.scenes[*i].text[*a..*b], &m.scenes[*j].text[*c..*d]))
            .collect::<Vec<_>>();
        match nli_with_identity(backend, &input) {
            Ok((output, identity)) => {
                model_identities.push(identity);
                cov.status = Status::Complete;
                cov.analyzed_units = output.len();
                for ((i, a, b, j, c, d), probs) in pairs.iter().zip(output) {
                    if probs[0] >= config.thresholds.nli_contradiction.unwrap_or(0.8)
                        && probs[0] > probs[1]
                        && probs[0] > probs[2]
                    {
                        let mut f = finding(
                            &m.document_id,
                            "NL-CONTINUITY-NLI",
                            "continuity",
                            Severity::Warning,
                            "Local NLI model identifies a possible contradiction.".into(),
                            "Review both passages, chronology, and speaker reliability.",
                            probs[0],
                            Scope::Passage {
                                scene_id: m.scenes[*j].id.clone(),
                            },
                            vec![
                                Evidence::new(
                                    &m.document_id,
                                    &m.scenes[*j].id,
                                    &m.scenes[*j].text,
                                    *c,
                                    *d,
                                    EvidenceRole::Primary,
                                )?,
                                Evidence::new(
                                    &m.document_id,
                                    &m.scenes[*i].id,
                                    &m.scenes[*i].text,
                                    *a,
                                    *b,
                                    EvidenceRole::Conflict,
                                )?,
                            ],
                        );
                        f.confidence = Confidence {
                            score: probs[0],
                            source: ConfidenceSource::Model,
                            method: format!(
                                "nli-3class-contradiction-neutral-entailment; {}",
                                config.model_revision
                            ),
                            calibration_id: None,
                        };
                        emit(f);
                    }
                }
                if pairs.len() < total_pairs {
                    cov.status = Status::HeuristicFallback;
                    cov.diagnostics.push(format!(
                        "candidate_limit; max_pairs={}, sentences_per_scene={}, window_scenes={}",
                        limits.nli_pairs, limits.nli_sentences_per_scene, limits.nli_window_scenes
                    ));
                }
            }
            Err(error) => {
                cov.status = match error.kind {
                    FailureKind::Unavailable => Status::Unavailable,
                    _ => Status::ExecutionFailure,
                };
                cov.diagnostics.push(error.code);
            }
        }
    }
    if oversized > 0 {
        cov.diagnostics
            .push(format!("model_input_too_long; {oversized} pairs skipped"));
    }
    if config.chronology {
        cov.status = Status::Unavailable;
    }
    if config.nli == ModelPolicy::Required && cov.status != Status::Complete {
        return Err(Error::AnalysisFailed(format!(
            "required NLI capability incomplete: {}",
            cov.diagnostics.join(", ")
        )));
    }
    if !disabled
        && config.nli == ModelPolicy::Fallback
        && matches!(cov.status, Status::Unavailable | Status::ExecutionFailure)
        && let Some(c) = coverage
            .iter_mut()
            .find(|c| c.analyzer == "continuity" && c.analyzed_units > 0)
    {
        c.status = Status::HeuristicFallback;
        c.diagnostics
            .push("optional_nli_unavailable_or_failed; explicit_state_heuristics_used".into());
    }
    coverage.push(cov);
    Ok(model_identities)
}

fn nli_with_identity(
    backend: &dyn InferenceBackend,
    pairs: &[(&str, &str)],
) -> std::result::Result<(Vec<Vec<f64>>, ModelIdentity), ModelFailure> {
    let generation = onnx::model_generation();
    let output = pairs
        .chunks(256)
        .map(|chunk| checked::nli(backend, chunk))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let identity = backend.model_identity(ModelKind::Nli)?;
    identity.validate(ModelKind::Nli)?;
    if onnx::model_generation() != generation {
        return Err(ModelFailure::execution("model_changed_during_inference"));
    }
    Ok((output.into_iter().flatten().collect(), identity))
}
