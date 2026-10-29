//! SHA-256 hex digests for pinned model manifests. Included with `#[path]`
//! by the tests that build manifests; not a test binary of its own.

use sha2::{Digest, Sha256};
use std::fmt::Write;

#[allow(dead_code)]
pub fn sha256_hex(bytes: impl AsRef<[u8]>) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}
