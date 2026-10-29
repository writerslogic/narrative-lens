use narrative_lens::substrate::text::count_syllables;
use std::{fs::File, io::Read};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: evaluate_pronunciation <prepared-word-inputs.json>")?;
    let mut bytes = Vec::new();
    File::open(path)?
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 16 * 1024 * 1024 {
        return Err("pronunciation inputs exceed16MiB".into());
    }
    let words: Vec<String> = serde_json::from_slice(&bytes)?;
    if words.len() > 200_000
        || words.iter().any(|word| {
            word.is_empty()
                || word.len() > 128
                || !word.bytes().all(|byte| byte.is_ascii_lowercase())
        })
    {
        return Err(
            "pronunciation inputs require at most200000 lowercase ASCII words of1..128bytes".into(),
        );
    }
    let results: Vec<_> = words
        .iter()
        .map(|word| (word, count_syllables(word)))
        .collect();
    println!("{}", serde_json::to_string(&results)?);
    Ok(())
}
