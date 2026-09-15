//! Timeline / chronology projection.
//!
//! A *read-only* projection of the manuscript for a "by timeline" view: the
//! documents in **manuscript order**, each annotated with the first explicit
//! temporal expression its text mentions (a year, a month, a clock time, a
//! weekday), or nothing when the prose names no time at all.
//!
//! v1 is deliberately shallow. We do **not** infer a story chronology — no
//! ordering by detected dates, no flashback/forward resolution, no relative
//! ("three days later") arithmetic. Manuscript order is the only order we trust,
//! because it is the one fact the writer actually authored; a chronology guessed
//! from scattered date mentions would reorder scenes the writer placed on
//! purpose (a cold-open, a framed flashback) and present that guess as truth.
//! The detected hint is therefore an *annotation on* the writer's order, never a
//! re-sort of it. Deeper temporal reasoning, if it ever ships, is a separate
//! opt-in feature layered on top of this honest baseline.
//!
//! This module is standalone — it has no dependency on any host application
//! state. It operates on `&[(document_id, text)]` so the detection and
//! ordering logic are unit-testable in isolation, and any caller holding the
//! document bodies can reuse it. Callers should supply the pairs in
//! manuscript order.

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// One manuscript document's place in the timeline view: its id, its zero-based
/// position in manuscript order, and the first explicit time expression found in
/// its text (if any).
///
/// `time_hint` is the *matched substring* (whitespace-normalized), e.g.
/// `"March 1847"` or `"9:30"`, and is `None` when the document names no
/// recognized time. It carries no parsed/normalized calendar value on purpose —
/// the core does no date math at this layer, so the host owns any
/// interpretation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimelineEntry {
    /// The document's stable id (as supplied by the caller).
    pub document_id: String,
    /// Zero-based position in manuscript order. This is the *only* ordering v1
    /// asserts; it is the input index, not a chronology inferred from hints.
    pub order: usize,
    /// The first detected explicit temporal expression in the document's text,
    /// whitespace-normalized and trimmed; `None` when no time is named.
    #[serde(default)]
    pub time_hint: Option<String>,
    #[serde(default)]
    pub chronology_key: Option<String>,
    #[serde(default)]
    pub lane: Option<String>,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
}

/// Writer-authored chronology metadata. It overlays conservative detection and
/// remains separate from prose, so changing a date or lane never rewrites text.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimelineOverride {
    pub document_id: String,
    #[serde(default)]
    pub chronology_key: Option<String>,
    #[serde(default)]
    pub lane: Option<String>,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub time_hint: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
}

/// One combined, case-insensitive regex for every temporal shape v1 recognizes,
/// compiled once. Built with `OnceLock` like the other cached regexes in the
/// crate, so the pattern is parsed a single time for the whole process, not per
/// document.
///
/// Conservatism is the whole design here — a false positive mislabels a scene's
/// time, which is worse than a missed one (the view simply shows no hint). So:
/// every alternative is `\b`-anchored (whole-word), the bare "May" branch is
/// excluded from the plain month list and only counts when an explicit day or
/// year follows it (otherwise the modal verb "may" would tag nearly every
/// document), and the standalone-year branch requires exactly four digits framed
/// by word boundaries.
///
/// The day/year tail of a month is written longest-alternative-first
/// (day+year, then year, then day). This matters because the `regex` crate has no
/// lookaround and uses leftmost-first (not leftmost-longest) alternation: a naive
/// optional `\s+\d{1,2}` would greedily eat "18" out of "March 1847" and stop,
/// leaving "47". Trying the day-and-year and year-only shapes before the
/// day-only shape makes "March 1847" match whole as a year.
fn temporal_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // `x` (free-spacing) mode: literal spaces/newlines in the pattern are
        // ignored, so the alternation reads top-to-bottom. `\s`/`\d` escapes and
        // the explicit `(?:\s+...)` groups still match real whitespace in input.
        regex::Regex::new(
            r"(?ix)
            # Month name (not bare 'May'), with an optional day and/or year.
            # Date tail tried longest-first: day+year, year-only, day-only.
            \b(?:january|february|march|april|june|july|august
                 |september|october|november|december)\b
              (?:\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}
                |\s+\d{4}
                |\s+\d{1,2}(?:st|nd|rd|th)?)?
            |
            # 'May' only when an explicit day or year follows (avoids the verb).
            \bmay\b
              (?:\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}
                |\s+\d{4}
                |\s+\d{1,2}(?:st|nd|rd|th)?)
            |
            # Clock time, 12- or 24-hour, e.g. 9:30 / 14:05.
            \b\d{1,2}:\d{2}\b
            |
            # Weekday markers.
            \b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b
            |
            # A bare four-digit year, e.g. 1847.
            \b\d{4}\b",
        )
        .expect("temporal-hint regex")
    })
}

/// The first explicit temporal expression in `text`, whitespace-normalized, or
/// `None` when none is found. Pure and deterministic: same text in, same hint
/// out. The match is normalized via `split_whitespace().join(" ")` so a hint that
/// straddles a line break ("March\n1847") returns as a clean "March 1847".
fn first_time_hint(text: &str) -> Option<String> {
    let m = temporal_re().find(text)?;
    let normalized = m.as_str().split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        None
    } else {
        Some(normalized)
    }
}

/// Build the timeline projection: the given documents, **in the order supplied**,
/// each tagged with its first detected time hint.
///
/// `order` is simply the input index — manuscript order is preserved exactly and
/// nothing is re-sorted (see the module docs for why v1 refuses to infer a
/// chronology). Pure and deterministic; empty input yields an empty Vec.
pub fn build_timeline(documents: &[(String, String)]) -> Vec<TimelineEntry> {
    documents
        .iter()
        .enumerate()
        .map(|(order, (document_id, text))| TimelineEntry {
            document_id: document_id.clone(),
            order,
            time_hint: first_time_hint(text),
            chronology_key: None,
            lane: None,
            label: None,
            notes: None,
        })
        .collect()
}

pub fn apply_overrides(entries: &mut [TimelineEntry], overrides: &[TimelineOverride]) {
    for entry in entries {
        if let Some(value) = overrides
            .iter()
            .find(|value| value.document_id == entry.document_id)
        {
            // Overlay semantics (see `TimelineOverride`'s doc comment): a field left
            // `None` in a partial override must leave the existing value alone, not
            // clear it.
            if value.chronology_key.is_some() {
                entry.chronology_key = value.chronology_key.clone();
            }
            if value.lane.is_some() {
                entry.lane = value.lane.clone();
            }
            if value.label.is_some() {
                entry.label = value.label.clone();
            }
            if value.notes.is_some() {
                entry.notes = value.notes.clone();
            }
            if value
                .time_hint
                .as_ref()
                .is_some_and(|hint| !hint.trim().is_empty())
            {
                entry.time_hint = value.time_hint.clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn docs(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(id, t)| (id.to_string(), t.to_string()))
            .collect()
    }

    #[test]
    fn month_and_year_is_detected_as_one_hint() {
        let tl = build_timeline(&docs(&[("d1", "It was March 1847. The frost held.")]));
        assert_eq!(tl.len(), 1);
        assert_eq!(tl[0].document_id, "d1");
        assert_eq!(tl[0].order, 0);
        assert_eq!(tl[0].time_hint.as_deref(), Some("March 1847"));
    }

    #[test]
    fn no_time_words_yields_none() {
        let tl = build_timeline(&docs(&[(
            "d1",
            "She walked the long road home, saying nothing.",
        )]));
        assert_eq!(tl[0].time_hint, None);
    }

    #[test]
    fn clock_time_is_detected() {
        let tl = build_timeline(&docs(&[("d1", "at 9:30 the bell rang")]));
        assert_eq!(tl[0].time_hint.as_deref(), Some("9:30"));
    }

    #[test]
    fn bare_year_is_detected() {
        let tl = build_timeline(&docs(&[("d1", "Everything changed in 1914, he said.")]));
        assert_eq!(tl[0].time_hint.as_deref(), Some("1914"));
    }

    #[test]
    fn weekday_is_detected_case_insensitively() {
        let tl = build_timeline(&docs(&[("d1", "By TUESDAY the letters had stopped.")]));
        assert_eq!(tl[0].time_hint.as_deref(), Some("TUESDAY"));
    }

    #[test]
    fn first_hint_wins_when_several_present() {
        // "Monday" precedes "9:30"; the earlier one in the text is returned.
        let tl = build_timeline(&docs(&[("d1", "On Monday at 9:30 it began.")]));
        assert_eq!(tl[0].time_hint.as_deref(), Some("Monday"));
    }

    #[test]
    fn bare_may_verb_is_not_a_false_positive() {
        // The modal verb "may" must not be mistaken for the month.
        let tl = build_timeline(&docs(&[("d1", "You may go now, she said quietly.")]));
        assert_eq!(tl[0].time_hint, None);
    }

    #[test]
    fn may_with_a_day_is_the_month() {
        let tl = build_timeline(&docs(&[("d1", "On May 5 the ship sailed.")]));
        assert_eq!(tl[0].time_hint.as_deref(), Some("May 5"));
    }

    #[test]
    fn hint_spanning_a_newline_is_normalized() {
        let tl = build_timeline(&docs(&[("d1", "It was March\n1847 when it ended.")]));
        assert_eq!(tl[0].time_hint.as_deref(), Some("March 1847"));
    }

    #[test]
    fn ordering_is_preserved_as_input_order() {
        let tl = build_timeline(&docs(&[
            ("ch1", "no time here"),
            ("ch2", "in 1920 the war was over"),
            ("ch3", "later still"),
        ]));
        assert_eq!(
            tl.iter()
                .map(|e| (e.document_id.as_str(), e.order))
                .collect::<Vec<_>>(),
            vec![("ch1", 0), ("ch2", 1), ("ch3", 2)]
        );
        assert_eq!(tl[1].time_hint.as_deref(), Some("1920"));
    }

    #[test]
    fn empty_input_yields_empty() {
        assert!(build_timeline(&[]).is_empty());
    }

    #[test]
    fn writer_override_adds_chronology_without_reordering_manuscript() {
        let mut entries = build_timeline(&docs(&[("d1", "Monday"), ("d2", "Tuesday")]));
        apply_overrides(
            &mut entries,
            &[TimelineOverride {
                document_id: "d2".into(),
                chronology_key: Some("1847-01-01".into()),
                lane: Some("Ada".into()),
                label: Some("Arrival".into()),
                time_hint: Some("New Year's Day".into()),
                notes: Some("Flashback".into()),
            }],
        );
        assert_eq!(entries[0].document_id, "d1");
        assert_eq!(entries[1].order, 1);
        assert_eq!(entries[1].time_hint.as_deref(), Some("New Year's Day"));
        assert_eq!(entries[1].lane.as_deref(), Some("Ada"));
    }

    #[test]
    fn partial_override_overlays_without_clearing_prior_fields() {
        let mut entries = build_timeline(&docs(&[("d1", "Monday")]));
        apply_overrides(
            &mut entries,
            &[TimelineOverride {
                document_id: "d1".into(),
                chronology_key: Some("1847-01-01".into()),
                lane: Some("Ada".into()),
                label: Some("Arrival".into()),
                time_hint: Some("New Year's Day".into()),
                notes: Some("Flashback".into()),
            }],
        );
        // A later override that only touches `label` must leave the rest alone.
        apply_overrides(
            &mut entries,
            &[TimelineOverride {
                document_id: "d1".into(),
                chronology_key: None,
                lane: None,
                label: Some("Departure".into()),
                time_hint: None,
                notes: None,
            }],
        );
        assert_eq!(entries[0].label.as_deref(), Some("Departure"));
        assert_eq!(entries[0].chronology_key.as_deref(), Some("1847-01-01"));
        assert_eq!(entries[0].lane.as_deref(), Some("Ada"));
        assert_eq!(entries[0].notes.as_deref(), Some("Flashback"));
    }

    #[test]
    fn entry_roundtrips_through_json() {
        let e = TimelineEntry {
            document_id: "d1".into(),
            order: 0,
            time_hint: Some("March 1847".into()),
            chronology_key: None,
            lane: None,
            label: None,
            notes: None,
        };
        let bytes = serde_json::to_vec(&e).unwrap();
        assert_eq!(serde_json::from_slice::<TimelineEntry>(&bytes).unwrap(), e);
        // time_hint defaults to None when absent from the wire.
        let bare: TimelineEntry =
            serde_json::from_str(r#"{"document_id":"d1","order":3}"#).unwrap();
        assert_eq!(bare.time_hint, None);
        assert_eq!(bare.order, 3);
    }
}
