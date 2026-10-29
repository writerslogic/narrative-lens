use narrative_lens::manuscript::*;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

const HELP: &str = r#"narrative-lens — local manuscript analysis

Usage:
  narrative-lens [OPTIONS] FILE.txt|FILE.md ...
  narrative-lens --manifest manuscript.json [OPTIONS]
  narrative-lens --list-rules [--format json]

Input and author controls:
  --manifest FILE             Stable scene IDs, paths, intentions and chronology
  --config FILE               Configuration JSON; overlays manifest configuration
  --document-id ID            Override manuscript identity (default: manuscript)
  --characters NAME,NAME       Known continuity subjects
  --rules ID,ID               Select rules; use --list-rules to inspect them
  --minimum-severity LEVEL    info|suggestion|warning|error
  --suppress ID               Suppress a rule; repeatable
  --model-root DIR            Explicit local NLI model root (requires onnx)
  --model-profile FILE        Pinned NLI model manifest; pair with --model-root

Reports and revision checks:
  --format terminal|json|sarif Output format (default: terminal)
  --baseline FILE             Previous Report or CLI JSON envelope
  --new-only                  Fail only on introduced findings; needs a baseline
  --fail-on LEVEL|none         Failure severity (default: error)
  --max-findings N            Allowed count at/above fail-on (default: 0)
  --require-complete          Fail on incomplete selected capabilities
  --verbose                   Include confidence provenance and all coverage
  -h, --help                  Show this help
  -V, --version               Show version

Each UTF-8 file is one scene; Markdown and newline sequences remain unchanged.
Manifest paths resolve relative to the manifest. JSON includes source URI mappings.
Exit: 0 below finding threshold; 1 threshold exceeded; 2 input/capability error or
an incomplete --new-only comparison. Reports go to stdout; errors go to stderr.
"#;
#[derive(Default)]
struct Args {
    format: String,
    config: Option<String>,
    manifest: Option<String>,
    baseline: Option<String>,
    rules: Option<String>,
    minimum: Option<Severity>,
    suppress: Vec<String>,
    fail: Option<Severity>,
    maximum: usize,
    new_only: bool,
    require_complete: bool,
    verbose: bool,
    list_rules: bool,
    document: Option<String>,
    files: Vec<String>,
    characters: Option<Vec<String>>,
    model_root: Option<String>,
    model_profile: Option<String>,
}
fn severity(value: &str) -> Result<Severity, String> {
    serde_json::from_value(Value::String(value.into()))
        .map_err(|_| "severity must be info, suggestion, warning or error".into())
}
fn parse() -> Result<Option<Args>, String> {
    let mut a = Args {
        format: "terminal".into(),
        fail: Some(Severity::Error),
        ..Default::default()
    };
    let mut input = env::args_os().skip(1).map(|v| {
        v.into_string()
            .map_err(|_| "arguments and paths must be valid Unicode".to_string())
    });
    while let Some(arg) = input.next() {
        let arg = arg?;
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{HELP}");
                return Ok(None);
            }
            "--version" | "-V" => {
                println!("narrative-lens {}", env!("CARGO_PKG_VERSION"));
                return Ok(None);
            }
            "--" => {
                a.files.extend(input.collect::<Result<Vec<_>, _>>()?);
                break;
            }
            "--new-only" => {
                a.new_only = true;
                continue;
            }
            "--require-complete" => {
                a.require_complete = true;
                continue;
            }
            "--verbose" => {
                a.verbose = true;
                continue;
            }
            "--list-rules" => {
                a.list_rules = true;
                continue;
            }
            _ => {}
        }
        if !arg.starts_with('-') {
            a.files.push(arg);
            continue;
        }
        let (option, inline) = arg
            .split_once('=')
            .map_or((arg.as_str(), None), |(k, v)| (k, Some(v.to_string())));
        if !matches!(
            option,
            "--format"
                | "--config"
                | "--manifest"
                | "--baseline"
                | "--characters"
                | "--rules"
                | "--minimum-severity"
                | "--suppress"
                | "--fail-on"
                | "--max-findings"
                | "--document-id"
                | "--model-root"
                | "--model-profile"
        ) {
            return Err(format!("unknown option {option}; use --help"));
        }
        let value = match inline {
            Some(value) => value,
            None => input
                .next()
                .ok_or_else(|| format!("{option} requires a value"))??,
        };
        if value.is_empty() || value.starts_with("--") {
            return Err(format!("{option} requires a value"));
        }
        match option {
            "--format" => a.format = value,
            "--config" => a.config = Some(value),
            "--manifest" => a.manifest = Some(value),
            "--baseline" => a.baseline = Some(value),
            "--model-root" => a.model_root = Some(value),
            "--model-profile" => a.model_profile = Some(value),
            "--characters" => {
                a.characters = Some(value.split(',').map(|s| s.trim().to_owned()).collect())
            }
            "--rules" => a.rules = Some(value),
            "--minimum-severity" => a.minimum = Some(severity(&value)?),
            "--suppress" => a.suppress.push(value),
            "--fail-on" => {
                a.fail = if value == "none" {
                    None
                } else {
                    Some(severity(&value)?)
                }
            }
            "--max-findings" => {
                a.maximum = value
                    .parse()
                    .map_err(|_| "--max-findings requires a nonnegative integer")?
            }
            "--document-id" => a.document = Some(value),
            _ => unreachable!(),
        }
    }
    if !matches!(a.format.as_str(), "terminal" | "json" | "sarif") {
        return Err("--format must be terminal, json or sarif".into());
    }
    if !a.list_rules && a.files.is_empty() && a.manifest.is_none() {
        return Err("provide text/Markdown files or --manifest; use --help".into());
    }
    if a.manifest.is_some() && !a.files.is_empty() {
        return Err("--manifest and positional manuscript files cannot be combined".into());
    }
    if a.new_only && a.baseline.is_none() {
        return Err("--new-only requires --baseline".into());
    }
    if a.model_root.is_some() != a.model_profile.is_some() {
        return Err("--model-root and --model-profile must be supplied together".into());
    }
    if a.model_root.is_some() && !cfg!(feature = "onnx") {
        return Err("model options require a CLI built with the onnx feature".into());
    }
    Ok(Some(a))
}
fn read_bounded(path: &Path, max: u64) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let file = fs::File::open(path)
        .map_err(|e| format!("cannot open {}: {}", path.display(), e.kind()))?;
    if !file
        .metadata()
        .map_err(|e| format!("cannot inspect {}: {}", path.display(), e.kind()))?
        .is_file()
    {
        return Err(format!("{} is not a regular file", path.display()));
    }
    let mut bytes = Vec::new();
    file.take(max + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e.kind()))?;
    if bytes.len() as u64 > max {
        return Err(format!(
            "{} exceeds the {max}-byte input limit",
            path.display()
        ));
    }
    Ok(bytes)
}
fn read_json(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&read_bounded(path, 64 * 1024 * 1024)?).map_err(|e| {
        format!(
            "invalid JSON in {} at line {}, column {}",
            path.display(),
            e.line(),
            e.column()
        )
    })
}
fn overlay(base: &mut Value, patch: Value) {
    match (base, patch) {
        (Value::Object(base), Value::Object(patch)) => {
            for (k, v) in patch {
                overlay(base.entry(k).or_insert(Value::Null), v);
            }
        }
        (base, patch) => *base = patch,
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Manifest {
    document_id: String,
    scenes: Vec<SceneFile>,
    #[serde(default)]
    known_characters: Vec<String>,
    #[serde(default)]
    configuration: Configuration,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct SceneFile {
    id: String,
    path: PathBuf,
    #[serde(default)]
    chronology: Option<Chronology>,
    #[serde(default)]
    chronology_lane: Option<String>,
    #[serde(default)]
    intentions: Vec<Intention>,
    #[serde(default)]
    purpose: String,
}
#[derive(Clone)]
struct Source {
    uri: String,
    path: PathBuf,
    lines: LineIndex,
}
#[derive(Clone)]
struct LineIndex {
    starts: Vec<(usize, u32)>,
    bom_utf16: u32,
}
impl LineIndex {
    fn new(text: &str) -> Self {
        let mut starts = vec![(0, 0)];
        let mut units = 0;
        let mut chars = text.char_indices().peekable();
        while let Some((i, ch)) = chars.next() {
            units += ch.len_utf16() as u32;
            if ch == '\r' {
                let end = if chars.peek().is_some_and(|(_, c)| *c == '\n') {
                    let (j, _) = chars.next().expect("peeked LF");
                    units += 1;
                    j + 1
                } else {
                    i + 1
                };
                starts.push((end, units));
            } else if ch == '\n' {
                starts.push((i + 1, units));
            }
        }
        Self {
            starts,
            bom_utf16: u32::from(text.starts_with('\u{feff}')),
        }
    }
    fn position(&self, byte: usize, utf16: u32) -> (usize, u32) {
        let row = self
            .starts
            .partition_point(|(b, _)| *b <= byte)
            .saturating_sub(1);
        (row + 1, utf16 - self.starts[row].1 + 1)
    }
}
fn encode_path(path: &str) -> String {
    let mut out = String::new();
    for b in path.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~/".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}
#[cfg(any(windows, test))]
fn windows_uri(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    let normalized = normalized
        .strip_prefix("//?/UNC/")
        .map(|s| format!("//{s}"))
        .unwrap_or_else(|| {
            normalized
                .strip_prefix("//?/")
                .unwrap_or(&normalized)
                .into()
        });
    if normalized.starts_with("//") {
        return format!("file:{}", encode_path(&normalized));
    }
    let mut encoded = encode_path(&normalized);
    if normalized.as_bytes().get(1) == Some(&b':') {
        encoded = encoded.replacen("%3A", ":", 1);
    }
    format!("file:///{encoded}")
}
fn source_uri(path: &Path) -> String {
    #[cfg(windows)]
    {
        windows_uri(&path.to_string_lossy())
    }
    #[cfg(not(windows))]
    {
        format!("file://{}", encode_path(&path.to_string_lossy()))
    }
}
fn inputs(a: &Args) -> Result<(Manuscript, BTreeMap<String, Source>), String> {
    let (mut m, files) = if let Some(path) = &a.manifest {
        let path = Path::new(path);
        let manifest:Manifest=serde_json::from_value(read_json(path)?).map_err(|_|format!("invalid manuscript manifest {}; expected documentId, scenes with id/path, and optional author controls",path.display()))?;
        if manifest.scenes.len() > 10_000 {
            return Err("manifest exceeds 10000 scenes".into());
        }
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        let mut scenes = Vec::new();
        let mut files = Vec::new();
        for f in manifest.scenes {
            let mut scene = Scene::new(f.id, "");
            scene.chronology = f.chronology;
            scene.chronology_lane = f.chronology_lane;
            scene.intentions = f.intentions;
            scene.purpose = f.purpose;
            files.push(base.join(f.path));
            scenes.push(scene);
        }
        (
            Manuscript {
                document_id: manifest.document_id,
                known_characters: manifest.known_characters,
                configuration: manifest.configuration,
                scenes,
            },
            files,
        )
    } else {
        if a.files.len() > 10_000 {
            return Err("input exceeds 10000 scenes".into());
        }
        let mut seen = BTreeSet::new();
        let mut files = Vec::new();
        let mut scenes = Vec::new();
        for file in &a.files {
            let path = fs::canonicalize(file)
                .map_err(|e| format!("cannot open {}: {}", file, e.kind()))?;
            if !seen.insert(path.clone()) {
                return Err(format!("duplicate manuscript file {file}"));
            }
            let cwd = fs::canonicalize(".")
                .map_err(|e| format!("cannot resolve working directory: {}", e.kind()))?;
            let id = encode_path(&path.strip_prefix(&cwd).unwrap_or(&path).to_string_lossy());
            if id.len() > 256 {
                return Err(
                    "path-derived scene ID exceeds 256 bytes; use a manifest with stable IDs"
                        .into(),
                );
            }
            scenes.push(Scene::new(id, ""));
            files.push(path);
        }
        (
            Manuscript {
                document_id: "manuscript".into(),
                known_characters: Vec::new(),
                configuration: Configuration::default(),
                scenes,
            },
            files,
        )
    };
    if let Some(id) = &a.document {
        m.document_id = id.clone();
    }
    if let Some(names) = &a.characters {
        m.known_characters = names.clone();
    }
    if let Some(path) = &a.config {
        let mut config =
            serde_json::to_value(&m.configuration).map_err(|_| "cannot construct configuration")?;
        overlay(&mut config, read_json(Path::new(path))?);
        m.configuration = serde_json::from_value(config)
            .map_err(|_| format!("invalid configuration fields or types in {path}"))?;
    }
    if let Some(rules) = &a.rules {
        m.configuration.rules = Some(rules.split(',').map(|r| r.trim().to_owned()).collect());
    }
    if let Some(level) = a.minimum {
        m.configuration.minimum_severity = level;
    }
    for rule in &a.suppress {
        m.configuration.suppressions.push(Suppression {
            rule_id: Some(rule.clone()),
            scene_id: None,
            fingerprint: None,
            reason: "CLI rule suppression".into(),
        });
    }
    m.validate().map_err(|e| e.to_string())?;
    let mut sources = BTreeMap::new();
    let mut bytes = 0usize;
    for (scene, path) in m.scenes.iter_mut().zip(files) {
        if !path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "txt" | "md" | "markdown"))
        {
            return Err(format!(
                "{} must be a text or Markdown file",
                path.display()
            ));
        }
        let content = read_bounded(&path, 16 * 1024 * 1024)?;
        bytes = bytes
            .checked_add(content.len())
            .ok_or("manuscript byte count overflow")?;
        if bytes > 64 * 1024 * 1024 {
            return Err("manuscript exceeds 64 MiB".into());
        }
        scene.text =
            String::from_utf8(content).map_err(|_| format!("{} is not UTF-8", path.display()))?;
        let path = fs::canonicalize(&path)
            .map_err(|e| format!("cannot resolve {}: {}", path.display(), e.kind()))?;
        sources.insert(
            scene.id.clone(),
            Source {
                uri: source_uri(&path),
                path,
                lines: LineIndex::new(&scene.text),
            },
        );
    }
    Ok((m, sources))
}
type PriorSources<'a> = (&'a BTreeMap<String, String>, &'a BTreeMap<String, u32>);

fn physical(
    e: &Evidence,
    sources: &BTreeMap<String, Source>,
    prior: Option<PriorSources<'_>>,
) -> Option<Value> {
    let source = sources.get(&e.scene_id);
    let (uri, bom_utf16) = if let Some((prior, boms)) = prior {
        (prior.get(&e.scene_id)?.clone(), *boms.get(&e.scene_id)?)
    } else {
        let source = source?;
        (source.uri.clone(), source.lines.bom_utf16)
    };
    let start = e.range.utf16_start.saturating_sub(bom_utf16);
    let end = e.range.utf16_end.saturating_sub(bom_utf16);
    let quote = if bom_utf16 > 0 && e.range.byte_start == 0 {
        e.quote.strip_prefix('\u{feff}').unwrap_or(&e.quote)
    } else {
        &e.quote
    };
    let mut region = json!({"charOffset":start,"charLength":end-start,"snippet":{"text":quote}});
    if prior.is_none()
        && let Some(source) = source
    {
        let (sl, sc) = source
            .lines
            .position(e.range.byte_start, e.range.utf16_start);
        let (el, ec) = source.lines.position(e.range.byte_end, e.range.utf16_end);
        region["startLine"] = json!(sl);
        region["startColumn"] = json!(if sl == 1 {
            sc.saturating_sub(bom_utf16).max(1)
        } else {
            sc
        });
        region["endLine"] = json!(el);
        region["endColumn"] = json!(if el == 1 {
            ec.saturating_sub(bom_utf16).max(1)
        } else {
            ec
        });
    }
    Some(json!({"artifactLocation":{"uri":uri},"region":region}))
}
fn sarif_result(
    f: &Finding,
    sources: &BTreeMap<String, Source>,
    prior: Option<PriorSources<'_>>,
) -> Value {
    let mut result = json!({"ruleId":f.rule_id,"level":sarif_level(f.severity),"message":{"text":f.message},"partialFingerprints":{"narrativeLens/v2":f.fingerprint},"properties":{"confidence":f.confidence,"scope":f.scope}});
    if let Some(e) = f.evidence.first() {
        if let Some(location) = physical(e, sources, prior) {
            result["locations"] = json!([{"physicalLocation":location}]);
        }
    } else if let Scope::Scene { scene_id } = &f.scope
        && let Some(source) = sources.get(scene_id)
    {
        result["locations"] = json!([{"physicalLocation":{"artifactLocation":{"uri":source.uri}}}]);
    }
    result["relatedLocations"]=json!(f.evidence.iter().skip(1).enumerate().filter_map(|(i,e)|physical(e,sources,prior).map(|physical|json!({"id":i+1,"physicalLocation":physical,"message":{"text":if e.role==EvidenceRole::Conflict{"Conflicting passage"}else{"Supporting context"}}}))).collect::<Vec<_>>());
    if result.get("locations").is_none() {
        result["properties"]["originalEvidence"] = json!(f.evidence);
    }
    result
}
fn sarif_level(level: Severity) -> &'static str {
    match level {
        Severity::Error => "error",
        Severity::Warning => "warning",
        _ => "note",
    }
}
fn sarif(
    report: &Report,
    sources: &BTreeMap<String, Source>,
    prior: &BTreeMap<String, String>,
    prior_boms: &BTreeMap<String, u32>,
    comparison: Option<&RevisionComparison>,
    execution_successful: bool,
) -> Value {
    let mut results = Vec::new();
    for f in &report.findings {
        let mut result = sarif_result(f, sources, None);
        if let Some(diff) = comparison {
            if diff
                .introduced
                .iter()
                .any(|n| n.fingerprint == f.fingerprint)
            {
                result["baselineState"] = json!("new");
            } else if let Some(p) = diff
                .persistent
                .iter()
                .find(|p| p.after.fingerprint == f.fingerprint)
            {
                result["baselineState"] = json!(if p.before.message == p.after.message
                    && p.before.severity == p.after.severity
                    && p.before.confidence == p.after.confidence
                {
                    "unchanged"
                } else {
                    "updated"
                });
                result["properties"]["revisionMatch"] = json!(p.match_kind);
            }
        }
        results.push(result);
    }
    for suppressed in &report.suppressed {
        let mut result = sarif_result(&suppressed.finding, sources, None);
        result["suppressions"] =
            json!([{"kind":"external","status":"accepted","justification":suppressed.reason}]);
        results.push(result);
    }
    if let Some(diff) = comparison {
        for f in &diff.resolved {
            if report
                .suppressed
                .iter()
                .any(|s| s.finding.fingerprint == f.fingerprint)
            {
                continue;
            }
            let mut result = sarif_result(f, sources, Some((prior, prior_boms)));
            result["baselineState"] = json!("absent");
            results.push(result);
        }
        for item in &diff.unassessed {
            if item.direction == ChangeDirection::Resolved {
                let mut result = sarif_result(&item.finding, sources, Some((prior, prior_boms)));
                result["kind"] = json!("review");
                result["level"] = json!("note");
                result["message"] = json!({"text":format!("Not rechecked: {} ({})",item.finding.message,item.reason)});
                result["properties"]["comparisonUnassessed"] = json!(true);
                results.push(result);
            }
        }
    }
    let rules=rule_definitions().into_iter().map(|r|json!({"id":r.id,"name":r.name,"shortDescription":{"text":r.name},"fullDescription":{"text":r.description},"defaultConfiguration":{"level":sarif_level(r.default_severity)},"properties":{"requiresModel":r.requires_model,"facadeAvailable":r.facade_available}})).collect::<Vec<_>>();
    json!({"$schema":"https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json","version":"2.1.0","runs":[{"tool":{"driver":{"name":"narrative-lens","version":env!("CARGO_PKG_VERSION"),"rules":rules}},"columnKind":"utf16CodeUnits","newlineSequences":["\r\n","\n","\r"],"invocations":[{"executionSuccessful":execution_successful}],"results":results,"properties":{"coverage":report.coverage,"diagnostics":report.diagnostics,"summary":report.summary}}]})
}
fn escaped_full(text: &str) -> String {
    text.chars().flat_map(char::escape_debug).collect()
}
fn escaped(text: &str) -> String {
    text.chars()
        .take(200)
        .flat_map(char::escape_debug)
        .collect()
}
fn terminal(
    out: &mut impl Write,
    report: &Report,
    sources: &BTreeMap<String, Source>,
    comparison: Option<&RevisionComparison>,
    verbose: bool,
) -> io::Result<()> {
    for f in &report.findings {
        let location = if let Some(e) = f.evidence.first() {
            if let Some(source) = sources.get(&e.scene_id) {
                let (line, column) = source
                    .lines
                    .position(e.range.byte_start, e.range.utf16_start);
                format!("{}:{line}:{column}", source.path.display())
            } else {
                e.scene_id.clone()
            }
        } else {
            match &f.scope {
                Scope::Scene { scene_id } | Scope::Passage { scene_id } => sources
                    .get(scene_id)
                    .map_or_else(|| scene_id.clone(), |s| s.path.display().to_string()),
                Scope::Manuscript { .. } => format!("{} (manuscript)", report.document_id),
            }
        };
        writeln!(
            out,
            "{}: {} {}: {}",
            escaped_full(&location),
            format!("{:?}", f.severity).to_lowercase(),
            f.rule_id,
            f.message
        )?;
        for e in &f.evidence {
            let label = match e.role {
                EvidenceRole::Primary => "passage",
                EvidenceRole::Conflict => "conflict",
                EvidenceRole::Context => "context",
            };
            if let Some(source) = sources.get(&e.scene_id) {
                let (line, column) = source
                    .lines
                    .position(e.range.byte_start, e.range.utf16_start);
                writeln!(
                    out,
                    "  {label} {}:{line}:{column}: {}",
                    escaped_full(&source.path.display().to_string()),
                    escaped(&e.quote)
                )?;
            }
        }
        if !f.suggestion.is_empty() {
            writeln!(out, "  suggestion: {}", escaped(&f.suggestion))?;
        }
        if verbose {
            writeln!(
                out,
                "  confidence: {:.2} ({:?}; {})",
                f.confidence.score,
                f.confidence.source,
                escaped(&f.confidence.method)
            )?;
        }
    }
    let active = report.summary.total_findings - report.summary.suppressed_findings;
    writeln!(
        out,
        "{active} findings; {} suppressed{}",
        report.summary.suppressed_findings,
        if report.summary.omitted_findings > 0 {
            format!("; {} omitted from display", report.summary.omitted_findings)
        } else {
            String::new()
        }
    )?;
    for c in &report.coverage {
        let intentional = c.status == Status::Unavailable
            && c.diagnostics.iter().any(|d| {
                d.contains("not_selected") || d.contains("disabled") || d.contains("lexicon_empty")
            });
        if verbose || c.status != Status::Complete && !intentional {
            writeln!(
                out,
                "  {}: {:?} ({}/{}) {}",
                c.analyzer,
                c.status,
                c.analyzed_units,
                c.total_units,
                c.diagnostics.join(", ")
            )?;
        }
    }
    if let Some(diff) = comparison {
        writeln!(
            out,
            "{} introduced, {} persistent, {} resolved, {} not rechecked",
            diff.introduced.len(),
            diff.persistent.len(),
            diff.resolved.len(),
            diff.unassessed.len()
        )?;
    }
    Ok(())
}
fn write_json(out: &mut impl Write, value: &Value) -> io::Result<()> {
    serde_json::to_writer_pretty(out, value)
        .map_err(|e| io::Error::new(e.io_error_kind().unwrap_or(io::ErrorKind::InvalidData), e))
}
fn run(a: Args) -> Result<u8, String> {
    let mut out = io::BufWriter::new(io::stdout().lock());
    if a.list_rules {
        let rules = rule_definitions();
        if a.format == "json" {
            write_json(&mut out, &json!(rules)).map_err(|_| "cannot write rule registry")?;
        } else {
            for r in rules {
                writeln!(
                    out,
                    "{}  {}{}",
                    r.id,
                    r.name,
                    if r.facade_available {
                        ""
                    } else {
                        " (adapter only)"
                    }
                )
                .map_err(|_| "cannot write rule registry")?;
            }
        }
        return Ok(0);
    }
    let (manuscript, sources) = inputs(&a)?;
    load_cli_model(&a, &manuscript.configuration)?;
    let report = analyze_manuscript(&manuscript).map_err(|e| e.to_string())?;
    report
        .validate_against(&manuscript)
        .map_err(|e| e.to_string())?;
    let mut prior = BTreeMap::new();
    let mut prior_boms: BTreeMap<String, u32> = BTreeMap::new();
    let mut comparison_complete = true;
    let comparison = if let Some(path) = &a.baseline {
        let mut value = read_json(Path::new(path))?;
        if let Some(source_map) = value.get_mut("sources") {
            prior = serde_json::from_value(source_map.take())
                .map_err(|_| "baseline has invalid source URI mappings")?;
        }
        if let Some(boms) = value.get_mut("sourceBomUtf16") {
            prior_boms = serde_json::from_value(boms.take())
                .map_err(|_| "baseline has invalid BOM mappings")?;
            if prior_boms.values().any(|&units| units > 1) {
                return Err("baseline BOM mappings must contain only 0 or 1".into());
            }
        }
        let baseline: Report =
            serde_json::from_value(value.get("report").cloned().unwrap_or(value))
                .map_err(|_| "invalid baseline Report")?;
        let comparison = try_compare_reports(&baseline, &report).map_err(|e| e.to_string())?;
        comparison_complete = comparison.is_complete(&baseline, &report);
        Some(comparison)
    } else {
        None
    };
    let count = if a.new_only {
        comparison
            .as_ref()
            .expect("baseline required")
            .introduced
            .iter()
            .filter(|f| a.fail.is_some_and(|s| f.severity >= s))
            .count()
    } else {
        report
            .summary
            .severity_counts
            .iter()
            .filter(|(s, _)| a.fail.is_some_and(|min| **s >= min))
            .map(|(_, n)| *n)
            .sum()
    };
    let incomplete = a.new_only && !comparison_complete
        || a.require_complete
            && report
                .coverage
                .iter()
                .any(|c| c.status != Status::Complete && !c.is_intentionally_disabled());
    let code = if incomplete {
        2
    } else if a.fail.is_some() && count > a.maximum {
        1
    } else {
        0
    };
    let source_uris = sources
        .iter()
        .map(|(id, s)| (id.clone(), s.uri.clone()))
        .collect::<BTreeMap<_, _>>();
    let source_boms = sources
        .iter()
        .map(|(id, source)| (id.clone(), source.lines.bom_utf16))
        .collect::<BTreeMap<_, _>>();
    let written = match a.format.as_str() {
        "json" => write_json(
            &mut out,
            &json!({"report":report,"comparison":comparison,"sources":source_uris,"sourceBomUtf16":source_boms}),
        ),
        "sarif" => write_json(
            &mut out,
            &sarif(
                &report,
                &sources,
                &prior,
                &prior_boms,
                comparison.as_ref(),
                !incomplete,
            ),
        ),
        _ => terminal(&mut out, &report, &sources, comparison.as_ref(), a.verbose),
    };
    match written
        .and_then(|_| writeln!(out))
        .and_then(|_| out.flush())
    {
        Ok(()) => {
            if incomplete {
                eprintln!(
                    "narrative-lens: requested complete analysis or revision comparison is incomplete; inspect coverage and unassessed findings"
                );
            }
            Ok(code)
        }
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(0),
        Err(e) => Err(format!("cannot write report: {}", e.kind())),
    }
}

fn load_cli_model(args: &Args, configuration: &Configuration) -> Result<(), String> {
    let (Some(directory), Some(profile)) = (&args.model_root, &args.model_profile) else {
        return Ok(());
    };
    if configuration.nli == ModelPolicy::Disabled || !configuration.selected("NL-CONTINUITY-NLI") {
        return Err("model options require selected NLI with nli=fallback or nli=required".into());
    }
    #[cfg(feature = "onnx")]
    {
        use narrative_lens::onnx::{self, checked::ModelKind, model::ModelManifest};
        let bytes = read_bounded(Path::new(profile), 16 * 1024)?;
        let manifest = ModelManifest::from_json(&bytes)
            .map_err(|e| format!("invalid model profile: {}", e.code))?;
        onnx::set_model_directory_verified(ModelKind::Nli, directory, &manifest)
            .map_err(|e| format!("NLI model unavailable: {}", e.code))?;
        Ok(())
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (directory, profile);
        Err("model options require the onnx feature".into())
    }
}
fn main() -> ExitCode {
    match parse().and_then(|a| a.map(run).transpose()) {
        Ok(Some(code)) => ExitCode::from(code),
        Ok(None) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("narrative-lens: {e}");
            ExitCode::from(2)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn windows_file_uris_preserve_drives_unc_and_unicode() {
        assert_eq!(
            windows_uri(r"C:\Books\Zoë story.md"),
            "file:///C:/Books/Zo%C3%AB%20story.md"
        );
        assert_eq!(windows_uri(r"\\?\C:\Books\a.md"), "file:///C:/Books/a.md");
        assert_eq!(
            windows_uri(r"\\server\share\a.md"),
            "file://server/share/a.md"
        );
        assert_eq!(
            windows_uri(r"\\?\UNC\server\share\a.md"),
            "file://server/share/a.md"
        );
    }
    #[test]
    fn source_positions_count_cr_lf_crlf_and_utf16() {
        let text = "a\rb\r\n😀x\ny";
        let index = LineIndex::new(text);
        assert_eq!(index.position(0, 0), (1, 1));
        assert_eq!(index.position(2, 2), (2, 1));
        assert_eq!(index.position(9, 7), (3, 3));
        assert_eq!(index.position(11, 9), (4, 1));
    }
}
