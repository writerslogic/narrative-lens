//! Node.js bindings (napi-rs), built with `--features node-api`.
//!
//! Deliberately a thin conversion layer, not `#[napi]` attributes on the core
//! analyzer types directly: the core types use `&'static str` grade labels and
//! internal enums that don't need to carry napi's derive machinery, and this
//! keeps the pure-Rust API (see `docs/ARCHITECTURE.md`'s design rule) free of
//! any Node-specific concern. Two conventions coexist: the 3 functions in this
//! file hand-write a `*Js` DTO + `From<CoreType>` impl per return type; the
//! `craft`/`structure_bindings`/`continuity`/`substrate` submodules use a
//! `serde_json::Value` JSON bridge instead, which scales to many-field nested
//! return types; explicit TypeScript overrides retain precise declarations. See
//! `docs/INTEGRATION.md` for what's covered and what's deliberately not.
//!
//! `#[allow(dead_code)]` on the submodule declarations below: every `#[napi]`
//! function's only caller is the FFI/JS boundary, which is invisible to
//! rustc's reachability analysis in the `cargo test` target (unlike a
//! `cdylib` build, where a `pub` item is implicitly exported). The 3
//! functions in this file don't need it because their own `#[cfg(test)]`
//! block below calls them directly.

use napi_derive::napi;

use crate::craft::{grammar, readability};
use crate::structure::structure;

#[allow(dead_code)]
mod continuity;
#[allow(dead_code)]
mod craft;
#[allow(dead_code)]
mod manuscript;
#[allow(dead_code)]
mod structure_bindings;
#[allow(dead_code)]
mod substrate;

pub(crate) fn reason(error: impl std::fmt::Display) -> napi::Error {
    napi::Error::from_reason(error.to_string())
}

pub(crate) fn decode_input<T: serde::de::DeserializeOwned>(
    value: serde_json::Value,
    message: &'static str,
) -> napi::Result<T> {
    serde_json::from_value(value).map_err(|_| napi::Error::from_reason(message))
}

pub(crate) fn encode_result<T: serde::Serialize>(value: T) -> napi::Result<serde_json::Value> {
    serde_json::to_value(value).map_err(reason)
}

fn unsigned_index(value: f64) -> napi::Result<usize> {
    if !value.is_finite() || value.fract() != 0.0 || !(0.0..=u32::MAX as f64).contains(&value) {
        return Err(napi::Error::from_reason(
            "index must be an unsigned 32-bit integer",
        ));
    }
    Ok(value as usize)
}

fn signed_integer(value: f64) -> napi::Result<i32> {
    if !value.is_finite()
        || value.fract() != 0.0
        || !(i32::MIN as f64..=i32::MAX as f64).contains(&value)
    {
        return Err(napi::Error::from_reason(
            "value must be a signed32-bit integer",
        ));
    }
    Ok(value as i32)
}

fn finite_range(value: f64, min: f64, max: f64) -> napi::Result<f64> {
    if !value.is_finite() || !(min..=max).contains(&value) {
        return Err(napi::Error::from_reason(
            "number is outside its supported finite range",
        ));
    }
    Ok(value)
}

fn matching_lengths(lengths: &[usize]) -> napi::Result<()> {
    if lengths
        .first()
        .is_some_and(|first| lengths.iter().any(|length| length != first))
    {
        return Err(napi::Error::from_reason(
            "parallel input arrays must have equal lengths",
        ));
    }
    Ok(())
}

fn bounded_texts<'a>(
    texts: impl IntoIterator<Item = &'a str>,
    max_items: usize,
) -> napi::Result<()> {
    let mut bytes = 0usize;
    for (index, text) in texts.into_iter().enumerate() {
        bytes = bytes
            .checked_add(text.len())
            .ok_or_else(|| napi::Error::from_reason("text byte limit exceeded"))?;
        if index >= max_items || text.len() > 16 * 1024 * 1024 || bytes > 64 * 1024 * 1024 {
            return Err(napi::Error::from_reason(
                "text inputs exceed their item limit,16MiB per item or64MiB total",
            ));
        }
    }
    Ok(())
}

#[napi(object)]
pub struct GradeLevelsJs {
    pub fkgl: String,
    pub gunning_fog: String,
    pub ari: String,
    pub coleman_liau: String,
    pub flesch_ease: String,
    pub dale_chall: String,
}

#[napi(object)]
pub struct ReadabilityResultJs {
    pub fkgl: f64,
    pub gunning_fog: f64,
    pub smog: f64,
    pub smog_available: bool,
    pub ari: f64,
    pub coleman_liau: f64,
    pub flesch_ease: f64,
    pub dale_chall: f64,
    pub sentence_count: u32,
    pub word_count: u32,
    pub syllable_count: u32,
    pub char_count: u32,
    pub polysyllable_count: u32,
    pub grade_levels: GradeLevelsJs,
    pub readability_variance: f64,
}

impl From<readability::ReadabilityResult> for ReadabilityResultJs {
    fn from(r: readability::ReadabilityResult) -> Self {
        ReadabilityResultJs {
            fkgl: r.fkgl,
            gunning_fog: r.gunning_fog,
            smog: r.smog,
            smog_available: r.smog_available,
            ari: r.ari,
            coleman_liau: r.coleman_liau,
            flesch_ease: r.flesch_ease,
            dale_chall: r.dale_chall,
            sentence_count: r.sentence_count as u32,
            word_count: r.word_count as u32,
            syllable_count: r.syllable_count as u32,
            char_count: r.char_count as u32,
            polysyllable_count: r.polysyllable_count as u32,
            grade_levels: GradeLevelsJs {
                fkgl: r.grade_levels.fkgl.to_string(),
                gunning_fog: r.grade_levels.gunning_fog.to_string(),
                ari: r.grade_levels.ari.to_string(),
                coleman_liau: r.grade_levels.coleman_liau.to_string(),
                flesch_ease: r.grade_levels.flesch_ease.to_string(),
                dale_chall: r.grade_levels.dale_chall.to_string(),
            },
            readability_variance: r.readability_variance,
        }
    }
}

/// Compute readability metrics (FKGL, Gunning Fog, SMOG, ARI, Coleman-Liau,
/// Flesch ease, Dale-Chall) for `text`.
#[napi(js_name = "computeReadability")]
pub fn compute_readability(text: String) -> napi::Result<ReadabilityResultJs> {
    bounded_texts([text.as_str()], 1)?;
    Ok(readability::compute_readability(&text).into())
}

#[napi(object)]
pub struct GrammarFindingJs {
    pub kind: String,
    pub utf16_start: u32,
    pub utf16_end: u32,
    pub message: String,
    pub replacements: Vec<String>,
}

impl From<grammar::GrammarFinding> for GrammarFindingJs {
    fn from(f: grammar::GrammarFinding) -> Self {
        let kind = match f.kind {
            grammar::FindingKind::Spelling => "spelling",
            grammar::FindingKind::Repetition => "repetition",
            grammar::FindingKind::Whitespace => "whitespace",
            grammar::FindingKind::Capitalization => "capitalization",
        };
        GrammarFindingJs {
            kind: kind.to_string(),
            utf16_start: f.utf16_start,
            utf16_end: f.utf16_end,
            message: f.message,
            replacements: f.replacements,
        }
    }
}

/// Run the built-in grammar checks (duplicated words, stray whitespace,
/// sentence-start capitalization) over `text`.
#[napi(js_name = "checkGrammar")]
pub fn check_grammar(text: String) -> napi::Result<Vec<GrammarFindingJs>> {
    bounded_texts([text.as_str()], 1)?;
    Ok(grammar::check(&text).into_iter().map(Into::into).collect())
}

/// List the names of the built-in narrative structure templates
/// (three_act, heros_journey, save_the_cat, kishotenketsu, nonlinear).
#[napi(js_name = "listStructureTemplates")]
pub fn list_structure_templates() -> Vec<String> {
    structure::list_templates()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_readability_wraps_the_core_result() {
        let js = compute_readability("The cat sat on the mat.".to_string()).unwrap();
        assert!(js.word_count > 0);
        assert!(!js.grade_levels.fkgl.is_empty());
    }

    #[test]
    fn check_grammar_maps_finding_kind_to_a_stable_string() {
        let findings = check_grammar("She was was tired.".to_string()).unwrap();
        assert!(findings.iter().any(|f| f.kind == "repetition"));
    }

    #[test]
    fn list_structure_templates_matches_the_core_list() {
        assert_eq!(list_structure_templates(), structure::list_templates());
    }
}
