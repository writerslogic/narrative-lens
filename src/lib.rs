//! narrative-lens: local, deterministic prose and narrative-craft analysis.
//!
//! Three analyzer categories, each a pure function -- text in, a structured,
//! confidence-scored result out. No project model, storage, or network access
//! lives here; that belongs in the consuming application.

pub mod continuity;
pub mod craft;
pub mod structure;
pub mod substrate;

pub mod onnx;

#[cfg(feature = "node-api")]
mod napi_bindings;

pub mod error;
pub mod types;

pub use error::{Error, Result};
pub use types::{AnalysisResult, Options};
