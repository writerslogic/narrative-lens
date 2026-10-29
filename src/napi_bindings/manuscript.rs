use crate::manuscript::{self, AnalysisContext, Manuscript, Report};
use napi::{Env, ScopedTask, bindgen_prelude::AsyncTask};
use napi_derive::napi;
use std::sync::{Arc, Mutex};

type ContextCell = Arc<Mutex<Option<AnalysisContext>>>;
fn error(e: impl std::fmt::Display) -> napi::Error {
    napi::Error::from_reason(e.to_string())
}
fn decode(value: serde_json::Value) -> napi::Result<Manuscript> {
    serde_json::from_value(value)
        .map_err(|_| error("invalid manuscript object; check documented fields, types, and limits"))
}
fn encode(value: impl serde::Serialize) -> napi::Result<serde_json::Value> {
    serde_json::to_value(value).map_err(error)
}
fn run(input: &Manuscript, cell: Option<&ContextCell>) -> napi::Result<Report> {
    if let Some(cell) = cell {
        let mut context = cell
            .lock()
            .map_err(|_| error("analysis context lock failed"))?
            .take()
            .ok_or_else(|| error("analysis context is busy"))?;
        let result = context.analyze(input).map_err(error);
        *cell
            .lock()
            .map_err(|_| error("analysis context lock failed"))? = Some(context);
        result
    } else {
        manuscript::analyze_manuscript(input).map_err(error)
    }
}
#[napi(
    js_name = "analyzeManuscript",
    ts_args_type = "manuscript: import('./manuscript').Manuscript",
    ts_return_type = "import('./manuscript').Report"
)]
pub fn analyze_manuscript(input: serde_json::Value) -> napi::Result<serde_json::Value> {
    encode(run(&decode(input)?, None)?)
}

pub struct AnalyzeTask {
    input: Manuscript,
    context: Option<ContextCell>,
}
impl<'task> ScopedTask<'task> for AnalyzeTask {
    type Output = serde_json::Value;
    type JsValue = napi::Unknown<'task>;
    fn compute(&mut self) -> napi::Result<Self::Output> {
        encode(run(&self.input, self.context.as_ref())?)
    }
    fn resolve(&mut self, env: &'task Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        env.to_js_value(&output)
    }
}
#[napi(
    js_name = "analyzeManuscriptAsync",
    ts_args_type = "manuscript: import('./manuscript').Manuscript",
    ts_return_type = "Promise<import('./manuscript').Report>"
)]
pub fn analyze_manuscript_async(input: serde_json::Value) -> napi::Result<AsyncTask<AnalyzeTask>> {
    Ok(AsyncTask::new(AnalyzeTask {
        input: decode(input)?,
        context: None,
    }))
}

#[napi(js_name = "ManuscriptContext")]
pub struct ManuscriptContext {
    context: ContextCell,
}
#[napi]
impl ManuscriptContext {
    #[napi(constructor)]
    pub fn new(max_cache_bytes: Option<u32>) -> napi::Result<Self> {
        Ok(Self {
            context: Arc::new(Mutex::new(Some(
                AnalysisContext::new(max_cache_bytes.unwrap_or(16 * 1024 * 1024) as usize)
                    .map_err(error)?,
            ))),
        })
    }
    #[napi(
        ts_args_type = "manuscript: import('./manuscript').Manuscript",
        ts_return_type = "import('./manuscript').Report"
    )]
    pub fn analyze(&self, input: serde_json::Value) -> napi::Result<serde_json::Value> {
        encode(run(&decode(input)?, Some(&self.context))?)
    }
    #[napi(
        ts_args_type = "manuscript: import('./manuscript').Manuscript",
        ts_return_type = "Promise<import('./manuscript').Report>"
    )]
    pub fn analyze_async(&self, input: serde_json::Value) -> napi::Result<AsyncTask<AnalyzeTask>> {
        Ok(AsyncTask::new(AnalyzeTask {
            input: decode(input)?,
            context: Some(Arc::clone(&self.context)),
        }))
    }
    #[napi(ts_return_type = "import('./manuscript').CacheStats")]
    pub fn stats(&self) -> napi::Result<serde_json::Value> {
        encode(
            self.context
                .lock()
                .map_err(|_| error("analysis context lock failed"))?
                .as_ref()
                .ok_or_else(|| error("analysis context is busy"))?
                .stats(),
        )
    }
    #[napi]
    pub fn clear(&self) -> napi::Result<()> {
        self.context
            .lock()
            .map_err(|_| error("analysis context lock failed"))?
            .as_mut()
            .ok_or_else(|| error("analysis context is busy"))?
            .clear();
        Ok(())
    }
}
#[napi(
    js_name = "compareManuscriptReports",
    ts_args_type = "before: import('./manuscript').Report, after: import('./manuscript').Report",
    ts_return_type = "import('./manuscript').RevisionComparison"
)]
pub fn compare_manuscript_reports(
    before: serde_json::Value,
    after: serde_json::Value,
) -> napi::Result<serde_json::Value> {
    let before: Report =
        serde_json::from_value(before).map_err(|_| error("invalid baseline report"))?;
    let after: Report =
        serde_json::from_value(after).map_err(|_| error("invalid current report"))?;
    encode(manuscript::try_compare_reports(&before, &after).map_err(error)?)
}

#[napi(
    js_name = "parseManuscriptReport",
    ts_args_type = "value: unknown",
    ts_return_type = "import('./manuscript').Report"
)]
pub fn parse_manuscript_report(value: serde_json::Value) -> napi::Result<serde_json::Value> {
    let report: Report =
        serde_json::from_value(value).map_err(|_| error("invalid report object"))?;
    report.validate().map_err(error)?;
    encode(report)
}
#[napi(
    js_name = "defaultManuscriptConfiguration",
    ts_return_type = "import('./manuscript').ResolvedConfiguration"
)]
pub fn default_manuscript_configuration() -> napi::Result<serde_json::Value> {
    encode(manuscript::Configuration::default())
}
#[napi(
    js_name = "listManuscriptRules",
    ts_return_type = "Array<import('./manuscript').RuleDefinition>"
)]
pub fn list_manuscript_rules() -> napi::Result<serde_json::Value> {
    encode(manuscript::rule_definitions())
}

pub struct LoadVerifiedModelTask {
    kind: crate::onnx::checked::ModelKind,
    directory: String,
    manifest: crate::onnx::model::ModelManifest,
}
impl<'task> ScopedTask<'task> for LoadVerifiedModelTask {
    type Output = serde_json::Value;
    type JsValue = napi::Unknown<'task>;
    fn compute(&mut self) -> napi::Result<Self::Output> {
        self.manifest.validate(self.kind).map_err(error)?;
        if self.directory.trim().is_empty()
            || self.directory.len() > 4096
            || self.directory.contains('\0')
        {
            return Err(error("invalid model directory"));
        }
        #[cfg(feature = "onnx")]
        {
            encode(
                crate::onnx::set_model_directory_verified(
                    self.kind,
                    &self.directory,
                    &self.manifest,
                )
                .map_err(error)?,
            )
        }
        #[cfg(not(feature = "onnx"))]
        {
            Err(error(
                "loadVerifiedModel requires an addon built with the onnx feature",
            ))
        }
    }
    fn resolve(&mut self, env: &'task Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        env.to_js_value(&output)
    }
}

#[napi(
    js_name = "loadVerifiedModel",
    ts_args_type = "kind: import('./manuscript.generated').ModelKind, modelRoot: string, manifest: import('./manuscript.generated').ModelManifest",
    ts_return_type = "Promise<import('./manuscript.generated').ModelIdentity>"
)]
pub fn load_verified_model(
    kind: String,
    model_root: String,
    manifest: serde_json::Value,
) -> napi::Result<AsyncTask<LoadVerifiedModelTask>> {
    let kind = serde_json::from_value(serde_json::Value::String(kind))
        .map_err(|_| error("unsupported model kind"))?;
    let manifest =
        serde_json::from_value(manifest).map_err(|_| error("invalid model manifest fields"))?;
    if model_root.trim().is_empty() || model_root.len() > 4096 || model_root.contains('\0') {
        return Err(error("invalid model directory"));
    }
    let directory = std::path::PathBuf::from(model_root);
    let directory = if directory.is_absolute() {
        directory
    } else {
        std::env::current_dir()
            .map_err(|_| error("cannot resolve model directory"))?
            .join(directory)
    };
    Ok(AsyncTask::new(LoadVerifiedModelTask {
        kind,
        directory: directory
            .to_str()
            .ok_or_else(|| error("model directory must be valid Unicode"))?
            .into(),
        manifest,
    }))
}
