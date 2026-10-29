use super::finding::Severity;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleMethod {
    ExactPattern,
    DictionaryMembership,
    FormulaWithHeuristicInputs,
    AuthorLexicon,
    TemplateHeuristic,
    ExplicitStateMachine,
    ModelProbability,
    AdapterOnly,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuleDefinition {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub analyzer: &'static str,
    pub default_severity: Severity,
    pub requires_model: bool,
    pub facade_available: bool,
    pub method: RuleMethod,
    pub limitation: &'static str,
}

macro_rules! registry {
    ($($variant:ident => ($id:literal, $name:literal, $description:literal, $analyzer:literal,
        $severity:ident, $method:ident, $limitation:literal)),+ $(,)?) => {
        #[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub enum RuleId {
            $(#[serde(rename = $id)] $variant),+
        }
        impl RuleId {
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];
            pub const fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $id),+ }
            }
            pub fn definition(self) -> RuleDefinition {
                match self { $(Self::$variant => {
                    let method = RuleMethod::$method;
                    RuleDefinition {
                        id: $id, name: $name, description: $description, analyzer: $analyzer,
                        default_severity: Severity::$severity,
                        requires_model: method == RuleMethod::ModelProbability,
                        facade_available: method != RuleMethod::AdapterOnly,
                        method, limitation: $limitation,
                    }
                }),+ }
            }
        }
        impl std::str::FromStr for RuleId {
            type Err = Error;
            fn from_str(value: &str) -> Result<Self> {
                match value { $($id => Ok(Self::$variant)),+, _ => Err(Error::InvalidInput("unknown rule ID".into())) }
            }
        }
        impl From<RuleId> for String {
            fn from(value: RuleId) -> Self { value.as_str().into() }
        }
        pub const RULES: &[&str] = &[$($id),+];
    };
}

registry! {
    GrammarRepetition => ("NL-GRAMMAR-REPETITION", "Repeated word",
        "Identical consecutive words separated only by whitespace.", "grammar", Warning, ExactPattern,
        "A literal repetition can be intentional; author controls can suppress it."),
    GrammarWhitespace => ("NL-GRAMMAR-WHITESPACE", "Extra spaces",
        "Consecutive interior ASCII spaces.", "grammar", Info, ExactPattern,
        "Spacing conventions and author formatting can be intentional."),
    GrammarSpelling => ("NL-GRAMMAR-SPELLING", "Dictionary absence",
        "Lowercase word absent from the local dictionary; names, dialect and invented words need author review.",
        "grammar", Suggestion, DictionaryMembership,
        "Dictionary absence is not proof of a spelling error; suggestions are lexical distances."),
    GrammarCapitalization => ("NL-GRAMMAR-CAPITALIZATION", "Capitalization adapter",
        "Adapter identity for legacy capitalization findings; the built-in checker does not emit it.",
        "grammar", Suggestion, AdapterOnly,
        "The manuscript facade rejects selection of this rule because it has no detector."),
    ReadabilityComplex => ("NL-READABILITY-COMPLEX", "Scene complexity",
        "Adequately sampled scene exceeds the configured maximum Flesch–Kincaid grade.",
        "readability", Suggestion, FormulaWithHeuristicInputs,
        "English segmentation and syllables are heuristic; formula grades are not reader comprehension measurements."),
    PacingLong => ("NL-PACING-LONG", "Long sentence average",
        "Adequately sampled scene exceeds the configured average sentence length.",
        "pacing", Suggestion, FormulaWithHeuristicInputs,
        "Sentence length is a defined surface measurement rather than a pacing-quality verdict."),
    ProseLexicon => ("NL-PROSE-LEXICON", "Author editorial term",
        "Whole-word occurrence in the author-configured editorial lexicon.",
        "prose", Suggestion, AuthorLexicon,
        "A source occurrence proves literal presence, not that the author's word choice is inappropriate."),
    StructureMissingBeat => ("NL-STRUCTURE-MISSING-BEAT", "Unmatched structure beat",
        "Template heuristic cannot match a beat; author-supplied scene purposes improve the available evidence.",
        "structure", Suggestion, TemplateHeuristic,
        "An unmatched heuristic template beat does not prove a missing narrative event."),
    ContinuityLocked => ("NL-CONTINUITY-LOCKED", "Constrained location",
        "Explicit subject changes location while an observed constraint remains active.",
        "continuity", Warning, ExplicitStateMachine,
        "Reasoning is limited to supported literal observations and chronology lanes, not all implied physical events."),
    ContinuityAttribute => ("NL-CONTINUITY-ATTRIBUTE", "Conflicting attribute",
        "Explicit assertions about a supported physical attribute disagree.",
        "continuity", Warning, ExplicitStateMachine,
        "Observations are bounded extraction patterns; metaphor, testimony and unobserved changes require review."),
    ContinuityObject => ("NL-CONTINUITY-OBJECT", "Object location",
        "Explicit supported-object locations disagree without an observed transfer.",
        "continuity", Warning, ExplicitStateMachine,
        "Absence of an observed transfer does not establish that no transfer occurred."),
    ContinuityNli => ("NL-CONTINUITY-NLI", "Model contradiction",
        "Optional local NLI model flags a sentence pair; score is an uncalibrated model probability.",
        "nli", Warning, ModelProbability,
        "Candidate retrieval is bounded; supplied model probabilities do not establish contradiction truth or calibration."),
}

pub fn rule_definitions() -> Vec<RuleDefinition> {
    RuleId::ALL.iter().map(|rule| rule.definition()).collect()
}
