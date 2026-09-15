//! Error type for narrative-lens operations.

use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// The caller passed input the analyzer can't operate on (empty text,
    /// out-of-range parameter, unknown enum label, etc).
    InvalidInput(String),
    /// The analyzer ran but could not produce a result (e.g. a dependency it
    /// needs internally failed).
    AnalysisFailed(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Error::AnalysisFailed(msg) => write!(f, "analysis failed: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
