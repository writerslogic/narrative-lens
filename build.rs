use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

fn main() {
    #[cfg(feature = "node-api")]
    napi_build::setup();

    println!("cargo:rerun-if-changed=src/substrate/freq_data.txt");

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("freq_table.rs");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let freq_path = Path::new(&manifest_dir).join("src/substrate/freq_data.txt");
    let input = match fs::File::open(&freq_path) {
        Ok(f) => f,
        Err(_) => {
            let mut out = fs::File::create(&dest_path).unwrap();
            writeln!(out, "static FREQ_WORDS: &[&str] = &[];").unwrap();
            writeln!(out, "static FREQ_VALUES: &[f32] = &[];").unwrap();
            return;
        }
    };

    let reader = BufReader::new(input);
    let mut entries: Vec<(String, f32)> = Vec::new();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() == 2 {
            let word = parts[0].to_string();
            if let Ok(freq) = parts[1].parse::<f32>() {
                if freq > 0.0 {
                    entries.push((word, freq));
                }
            }
        }
    }

    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut out = fs::File::create(&dest_path).unwrap();
    writeln!(out, "static FREQ_WORDS: &[&str] = &[").unwrap();
    for (word, _) in &entries {
        writeln!(
            out,
            "    \"{}\",",
            word.replace('\\', "\\\\").replace('"', "\\\"")
        )
        .unwrap();
    }
    writeln!(out, "];").unwrap();
    writeln!(out, "#[allow(clippy::approx_constant)]").unwrap();
    writeln!(out, "static FREQ_VALUES: &[f32] = &[").unwrap();
    for (_, freq) in &entries {
        writeln!(out, "    {:.2},", freq).unwrap();
    }
    writeln!(out, "];").unwrap();

    eprintln!(
        "cargo:warning=Generated frequency table with {} entries",
        entries.len()
    );
}
