//! Top-level result/options types. Reserved for a future unified facade over
//! the individual analyzer functions; each analyzer currently exposes its own
//! function-specific result and options types, and nothing in this crate
//! depends on these yet.

/// Placeholder for shared analyzer options. Not yet used by any analyzer.
#[derive(Debug, Clone, Default)]
pub struct Options {}

/// Placeholder for a unified analysis result. Not yet used by any analyzer.
#[derive(Debug, Clone, Default)]
pub struct AnalysisResult {}
