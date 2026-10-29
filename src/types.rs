//! Legacy empty placeholders, retained so existing imports and struct literals
//! keep compiling. The optional facade uses `manuscript::Configuration` and
//! `manuscript::Report`; standalone analyzers retain their own types.

/// Placeholder for shared analyzer options. Not yet used by any analyzer.
#[derive(Debug, Clone, Default)]
pub struct Options {}

/// Placeholder for a unified analysis result. Not yet used by any analyzer.
#[derive(Debug, Clone, Default)]
pub struct AnalysisResult {}
