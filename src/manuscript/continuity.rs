use super::{
    Coverage, Diagnostic, Manuscript, Scene, Status,
    config::Intention,
    finding::{Confidence, Evidence, EvidenceRole, Finding, Scope, Severity},
};
use crate::{Result, continuity::entity::CorrectionAction};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EventSelection {
    pub location: bool,
    pub attribute: bool,
    pub object: bool,
    pub knowledge: bool,
}
impl EventSelection {
    pub fn enabled(&self) -> bool {
        self.location || self.attribute || self.object || self.knowledge
    }
    fn includes(&self, kind: &EventKind) -> bool {
        match kind {
            EventKind::Location(_) | EventKind::Lock | EventKind::Release => self.location,
            EventKind::Attribute(_) => self.attribute,
            EventKind::Object(..) | EventKind::ObjectTransfer(_) => self.object,
            EventKind::Knowledge(_) => self.knowledge,
        }
    }
}
#[derive(Debug, Clone)]
pub(crate) struct Event {
    pub start: usize,
    pub end: usize,
    pub character: String,
    pub kind: EventKind,
    pub range: Option<super::TextRange>,
}
#[derive(Debug, Clone)]
pub(crate) enum EventKind {
    Location(String),
    Lock,
    Release,
    Attribute(String),
    Object(String, String),
    ObjectTransfer(String),
    Knowledge(String),
}

#[derive(Debug)]
pub(crate) struct Parser {
    aliases: Vec<(String, String)>,
    re: Option<Regex>,
    transfer: Option<Regex>,
}
impl Parser {
    pub(crate) fn new(
        characters: &[String],
        corrections: &[crate::continuity::entity::EntityCorrection],
    ) -> Result<Self> {
        let mut names: BTreeMap<String, String> =
            characters.iter().map(|c| (c.clone(), c.clone())).collect();
        for c in corrections {
            match &c.action {
                CorrectionAction::Assign { canonical } => {
                    names.insert(c.surface.clone(), canonical.clone());
                }
                CorrectionAction::Reject => {
                    names.remove(&c.surface);
                }
            }
        }
        let mut ordered: Vec<_> = names.iter().collect();
        ordered.sort_by_key(|(surface, _)| std::cmp::Reverse(surface.len()));
        let aliases: Vec<_> = ordered
            .iter()
            .enumerate()
            .map(|(i, (_, canonical))| (format!("alias{i}"), (*canonical).clone()))
            .collect();
        let re = if !names.is_empty() {
            let cp = ordered
                .iter()
                .enumerate()
                .map(|(i, (surface, _))| format!("(?P<alias{i}>{})", regex::escape(surface)))
                .collect::<Vec<_>>()
                .join("|");
            let pattern = format!(
                r#"(?i)^[^\p{{L}}\p{{N}}\x22“‘«]*(?P<who>{cp})\s+(?:(?P<verb>was locked in|was trapped in|was in|was at|entered|walked into|escaped from|was released from)\s+(?:the\s+)?(?P<place>[\p{{L}}\p{{N}}][\p{{L}}\p{{N}}\s-]{{0,79}}?)[.!;]|(?:had|has)\s+(?P<eyes>blue|brown|green|gray|grey|hazel|black|amber|violet|red)\s+eyes[.!;]|(?:discovered|learned|knew|realized)\s+that\s+(?P<info>[^.!;]{{1,256}})[.!;])"#
            );
            Some(Regex::new(&pattern).map_err(|_| {
                crate::Error::InvalidInput("character matcher exceeds regex limits".into())
            })?)
        } else {
            None
        };
        let transfer = if names.is_empty() {
            None
        } else {
            let cp = names
                .keys()
                .map(|s| regex::escape(s))
                .collect::<Vec<_>>()
                .join("|");
            Some(Regex::new(&format!(r"(?i)^[^\p{{L}}\p{{N}}\x22“‘«]*(?:{cp})\s+(?:moved|carried|took|picked up|gave|handed)\s+(?:the\s+)?(key|sword|letter|ring|book)\b[^.!;]{{0,256}}[.!;]")).map_err(|_| crate::Error::InvalidInput("transfer matcher exceeds regex limits".into()))?)
        };
        Ok(Self {
            aliases,
            re,
            transfer,
        })
    }
    pub(crate) fn extract_bounded(
        &self,
        text: &str,
        limit: usize,
        selection: EventSelection,
    ) -> Vec<Event> {
        let mut events = Vec::new();
        if let Some(re) = &self.re {
            for (start, end) in crate::substrate::utils::sentence_spans(text) {
                if events.len() > limit {
                    break;
                }
                let sentence = &text[start..end];
                if matches!(sentence.trim_start().chars().next(), Some('"' | '“' | '‘')) {
                    continue;
                }
                if let Some(caps) = re.captures(sentence) {
                    let m = caps.get(0).expect("capture has whole match");
                    let character = self
                        .aliases
                        .iter()
                        .find(|(alias, _)| caps.name(alias).is_some())
                        .map(|(_, c)| c.clone())
                        .expect("captured known alias");
                    let mut push = |kind| {
                        if !selection.includes(&kind) {
                            return;
                        }
                        events.push(Event {
                            range: None,
                            start: start + m.start(),
                            end: start + m.end(),
                            character: character.clone(),
                            kind,
                        })
                    };
                    if let Some(place) = caps.name("place") {
                        let verb = caps["verb"].to_lowercase();
                        if verb == "escaped from" || verb == "was released from" {
                            push(EventKind::Release);
                        } else {
                            push(EventKind::Location(normalize(place.as_str())));
                            if verb.contains("locked") || verb.contains("trapped") {
                                push(EventKind::Lock);
                            }
                        }
                    } else if let Some(eyes) = caps.name("eyes") {
                        push(EventKind::Attribute(
                            if eyes.as_str().eq_ignore_ascii_case("grey") {
                                "gray".into()
                            } else {
                                normalize(eyes.as_str())
                            },
                        ));
                    } else if let Some(info) = caps.name("info") {
                        push(EventKind::Knowledge(normalize(info.as_str())));
                    }
                }
                // Reuse the legacy lock vocabulary, but only an explicit known
                // subject/location in this scene can acquire a constraint.
                if selection.location
                    && let Some(caps) = door_regex().captures(sentence)
                {
                    let release = caps[1].to_lowercase().starts_with("un");
                    events.push(Event {
                        start,
                        end,
                        character: String::new(),
                        range: None,
                        kind: if release {
                            EventKind::Release
                        } else {
                            EventKind::Lock
                        },
                    });
                }
            }
        }
        if selection.object
            && let Some(re) = &self.transfer
        {
            for (start, end) in crate::substrate::utils::sentence_spans(text) {
                if events.len() > limit {
                    break;
                }
                if let Some(c) = re.captures(&text[start..end]) {
                    events.push(Event {
                        start,
                        end,
                        character: String::new(),
                        range: None,
                        kind: EventKind::ObjectTransfer(normalize(&c[1])),
                    });
                }
            }
        }
        let object = object_regex();
        for (start, end) in crate::substrate::utils::sentence_spans(text) {
            if events.len() > limit {
                break;
            }
            if selection.object
                && let Some(c) = object.captures(&text[start..end])
            {
                let m = c.get(0).expect("capture has whole match");
                events.push(Event {
                    start: start + m.start(),
                    end: start + m.end(),
                    character: String::new(),
                    range: None,
                    kind: EventKind::Object(normalize(&c[1]), normalize(&c[2])),
                });
            }
        }
        if !events.is_empty() {
            let offsets = crate::substrate::offsets::OffsetIndex::new(text)
                .expect("scene fits offset coordinates");
            for e in &mut events {
                e.range = Some(super::TextRange {
                    byte_start: e.start,
                    byte_end: e.end,
                    utf16_start: offsets.utf16(e.start).expect("event start boundary"),
                    utf16_end: offsets.utf16(e.end).expect("event end boundary"),
                });
            }
        }
        events.sort_by_key(|e| (e.start, e.end));
        events
    }
}
fn door_regex() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)^\s*(?:someone|a guard|the guard)\s+(locked|unlocked|bolted|unbolted|barred|unbarred|sealed|unsealed)\s+the\s+(?:door|gate|exit)[.!;]\s*$").expect("door regex"))
}
fn object_regex() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)^\s*The\s+(key|sword|letter|ring|book)\s+was\s+(?:in|on|at)\s+(?:the\s+)?([\p{L}\p{N}][\p{L}\p{N}\s-]{0,79}?)[.!;]").expect("object regex"))
}
fn normalize(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReaderKnowledge {
    pub after_scene_id: String,
    pub introduced: Vec<Evidence>,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChronologyReport {
    pub lanes: BTreeMap<String, Vec<String>>,
    pub ambiguous_scene_ids: Vec<String>,
    pub reader_knowledge: Vec<ReaderKnowledge>,
}

fn evidence(manuscript: &Manuscript, scene: &Scene, event: &Event, role: EvidenceRole) -> Evidence {
    let range = event
        .range
        .as_ref()
        .expect("indexed original event")
        .clone();
    Evidence {
        document_id: manuscript.document_id.clone(),
        scene_id: scene.id.clone(),
        quote: scene.text[event.start..event.end].into(),
        role,
        range,
        context: Some(super::finding::context(&scene.text, event.start, event.end)),
    }
}
fn conflict(
    rule: &str,
    message: String,
    current: Evidence,
    prior: &Evidence,
    score: f64,
) -> Finding {
    let scene_id = current.scene_id.clone();
    let mut prior = prior.clone();
    prior.role = EvidenceRole::Conflict;
    Finding {
        document_id: current.document_id.clone(),
        rule_id: rule.into(),
        analyzer: "continuity".into(),
        severity: Severity::Warning,
        message,
        suggestion: "Check both passages; show the transition or record the intended exception."
            .into(),
        confidence: Confidence::heuristic(
            score,
            "explicit-state-v1; absence of an observed transition is not proof of an error",
        ),
        scope: Scope::Passage { scene_id },
        evidence: vec![current, prior],
        fingerprint: String::new(),
    }
}

pub(crate) fn analyze(
    manuscript: &Manuscript,
    events: &[&[Event]],
    limited: &[bool],
    emit: &mut dyn FnMut(Finding),
    diagnostics: &mut Vec<Diagnostic>,
) -> (ChronologyReport, Coverage) {
    let mut report = ChronologyReport::default();
    let mut seen = BTreeSet::new();
    for (scene, events) in manuscript.scenes.iter().zip(events) {
        let mut introduced = Vec::new();
        for e in *events {
            if let EventKind::Knowledge(info) = &e.kind {
                // Reader knowledge follows presentation even when the scene is a flashback.
                if seen.insert(info.clone()) {
                    introduced.push(evidence(manuscript, scene, e, EvidenceRole::Context));
                }
            }
        }
        report.reader_knowledge.push(ReaderKnowledge {
            after_scene_id: scene.id.clone(),
            introduced,
        });
    }
    let mut lanes: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    if manuscript.configuration.chronology {
        let mut keys = BTreeMap::<(String, String), Vec<usize>>::new();
        let mut missing_lanes = BTreeSet::new();
        let mut unknown_lane = false;
        for (i, scene) in manuscript.scenes.iter().enumerate() {
            if let Some(c) = &scene.chronology {
                keys.entry((c.lane.clone(), c.key.clone()))
                    .or_default()
                    .push(i);
            } else {
                report.ambiguous_scene_ids.push(scene.id.clone());
                if let Some(lane) = &scene.chronology_lane {
                    missing_lanes.insert(lane.clone());
                } else {
                    unknown_lane = true;
                }
            }
        }
        if unknown_lane {
            keys.clear();
            report.ambiguous_scene_ids = manuscript.scenes.iter().map(|s| s.id.clone()).collect();
        }
        for ((lane, _), indices) in keys {
            if missing_lanes.contains(&lane) {
                report
                    .ambiguous_scene_ids
                    .extend(indices.iter().map(|i| manuscript.scenes[*i].id.clone()));
                continue;
            }
            if indices.len() > 1 {
                report
                    .ambiguous_scene_ids
                    .extend(indices.iter().map(|i| manuscript.scenes[*i].id.clone()));
            }
            lanes.entry(lane).or_default().extend(indices);
        }
        if !report.ambiguous_scene_ids.is_empty() {
            diagnostics.push(Diagnostic { code: "chronology_ambiguous".into(), message: "Missing keys prevent comparisons in the affected lane, or across all lanes when the lane is unknown; ties form local uncertainty barriers. No order or cross-lane state is inferred.".into(), scene_ids: report.ambiguous_scene_ids.clone() });
        }
    } else {
        lanes.insert(
            "presentation".into(),
            (0..manuscript.scenes.len()).collect(),
        );
    }
    let mut analyzed = 0;
    for (lane, indices) in lanes {
        report.lanes.insert(
            lane,
            indices
                .iter()
                .map(|i| manuscript.scenes[*i].id.clone())
                .collect(),
        );
        let mut locations = BTreeMap::<String, (String, Evidence)>::new();
        let mut locks = BTreeMap::<String, (String, Evidence, Evidence)>::new();
        let mut attributes = BTreeMap::<String, (String, Evidence)>::new();
        let mut objects = BTreeMap::<String, (String, Evidence)>::new();
        for i in indices {
            let scene = &manuscript.scenes[i];
            if limited[i]
                || report.ambiguous_scene_ids.contains(&scene.id)
                || scene.intentions.contains(&Intention::Supernatural)
            {
                // An exception can change arbitrary physical state; retaining it
                // would make later state comparisons falsely authoritative.
                locations.clear();
                locks.clear();
                attributes.clear();
                objects.clear();
                continue;
            }
            if scene.text.trim().is_empty() {
                continue;
            }
            analyzed += 1;
            let mut present = BTreeSet::new();
            for event in events[i] {
                let current = evidence(manuscript, scene, event, EvidenceRole::Primary);
                match &event.kind {
                    EventKind::Location(place) => {
                        if let Some((locked, prior, constraint)) = locks.get(&event.character)
                            && locked != place
                        {
                            let mut finding = conflict(
                                "NL-CONTINUITY-LOCKED",
                                format!(
                                    "{} moves from a constrained location without an observed release.",
                                    event.character
                                ),
                                current.clone(),
                                prior,
                                0.8,
                            );
                            if constraint.range != prior.range
                                || constraint.scene_id != prior.scene_id
                            {
                                let mut supporting = constraint.clone();
                                supporting.role = EvidenceRole::Context;
                                finding.evidence.push(supporting);
                            }
                            emit(finding);
                        }
                        locations.insert(event.character.clone(), (place.clone(), current.clone()));
                        present.insert(event.character.clone());
                    }
                    EventKind::Lock => {
                        let targets = if event.character.is_empty() {
                            let places = present
                                .iter()
                                .filter_map(|c| locations.get(c).map(|(place, _)| place))
                                .collect::<BTreeSet<_>>();
                            if places.len() > 1 {
                                diagnostics.push(Diagnostic{code:"ambiguous_constraint".into(),message:"An unscoped door constraint cannot identify which of several locations it affects.".into(),scene_ids:vec![scene.id.clone()]});
                                Vec::new()
                            } else {
                                present.iter().cloned().collect::<Vec<_>>()
                            }
                        } else {
                            vec![event.character.clone()]
                        };
                        for c in targets {
                            if let Some((place, location)) = locations.get(&c) {
                                locks.insert(c, (place.clone(), location.clone(), current.clone()));
                            }
                        }
                    }
                    EventKind::Release => {
                        if event.character.is_empty() {
                            if locks
                                .values()
                                .map(|(place, _, _)| place)
                                .collect::<BTreeSet<_>>()
                                .len()
                                > 1
                            {
                                diagnostics.push(Diagnostic{code:"ambiguous_constraint".into(),message:"An unscoped release makes several active constraints uncertain.".into(),scene_ids:vec![scene.id.clone()]});
                            }
                            locks.clear();
                        } else {
                            locks.remove(&event.character);
                        }
                    }
                    EventKind::Attribute(value) => {
                        if let Some((before, prior)) = attributes.get(&event.character)
                            && before != value
                        {
                            emit(conflict(
                                "NL-CONTINUITY-ATTRIBUTE",
                                format!("{} has conflicting eye colors.", event.character),
                                current.clone(),
                                prior,
                                0.8,
                            ));
                        }
                        attributes.insert(event.character.clone(), (value.clone(), current));
                    }
                    EventKind::Object(object, place) => {
                        if let Some((before, prior)) = objects.get(object)
                            && before != place
                        {
                            emit(conflict(
                                "NL-CONTINUITY-OBJECT",
                                format!(
                                    "The {object} changes location without an observed transfer."
                                ),
                                current.clone(),
                                prior,
                                0.5,
                            ));
                        }
                        objects.insert(object.clone(), (place.clone(), current));
                    }
                    EventKind::ObjectTransfer(object) => {
                        objects.remove(object);
                    }
                    EventKind::Knowledge(_) => {}
                }
            }
        }
    }
    let eligible = manuscript
        .scenes
        .iter()
        .filter(|s| !s.text.trim().is_empty())
        .count();
    (report, Coverage { analyzer: "continuity".into(), status: if eligible == 0 { Status::InsufficientInput } else if analyzed < eligible { Status::HeuristicFallback } else { Status::Complete }, total_units: manuscript.scenes.len(), analyzed_units: analyzed, unit: "scenes".into(), diagnostics: vec!["Explicit subject/location, eye-color, and object-location patterns only; unrecognized facts remain unknown.".into()] })
}
