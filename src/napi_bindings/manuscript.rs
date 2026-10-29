use crate::manuscript::{self, AnalysisContext, Manuscript, Report};
use napi::{Env, ScopedTask, bindgen_prelude::AsyncTask};
use napi_derive::napi;
use std::sync::{Arc, Mutex};

type ContextCell = Arc<Mutex<Option<AnalysisContext>>>;
fn decode(value: serde_json::Value) -> napi::Result<Manuscript> {
    super::decode_input(
        value,
        "invalid manuscript object; check documented fields, types, and limits",
    )
}
fn run(input: &Manuscript, cell: Option<&ContextCell>) -> napi::Result<Report> {
    if let Some(cell) = cell {
        let mut context = cell
            .lock()
            .map_err(|_| super::reason("analysis context lock failed"))?
            .take()
            .ok_or_else(|| super::reason("analysis context is busy"))?;
        let result = context.analyze(input).map_err(super::reason);
        *cell
            .lock()
            .map_err(|_| super::reason("analysis context lock failed"))? = Some(context);
        result
    } else {
        manuscript::analyze_manuscript(input).map_err(super::reason)
    }
}
#[napi(
    js_name = "analyzeManuscript",
    ts_args_type = "manuscript: import('./manuscript').Manuscript",
    ts_return_type = "import('./manuscript').Report"
)]
pub fn analyze_manuscript(input: serde_json::Value) -> napi::Result<serde_json::Value> {
    super::encode_result(run(&decode(input)?, None)?)
}

pub struct AnalyzeTask {
    input: Manuscript,
    context: Option<ContextCell>,
}
impl<'task> ScopedTask<'task> for AnalyzeTask {
    type Output = serde_json::Value;
    type JsValue = napi::Unknown<'task>;
    fn compute(&mut self) -> napi::Result<Self::Output> {
        super::encode_result(run(&self.input, self.context.as_ref())?)
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
                    .map_err(super::reason)?,
            ))),
        })
    }
    #[napi(
        ts_args_type = "manuscript: import('./manuscript').Manuscript",
        ts_return_type = "import('./manuscript').Report"
    )]
    pub fn analyze(&self, input: serde_json::Value) -> napi::Result<serde_json::Value> {
        super::encode_result(run(&decode(input)?, Some(&self.context))?)
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
        super::encode_result(
            self.context
                .lock()
                .map_err(|_| super::reason("analysis context lock failed"))?
                .as_ref()
                .ok_or_else(|| super::reason("analysis context is busy"))?
                .stats(),
        )
    }
    #[napi]
    pub fn clear(&self) -> napi::Result<()> {
        self.context
            .lock()
            .map_err(|_| super::reason("analysis context lock failed"))?
            .as_mut()
            .ok_or_else(|| super::reason("analysis context is busy"))?
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
    let before: Report = super::decode_input(before, "invalid baseline report")?;
    let after: Report = super::decode_input(after, "invalid current report")?;
    super::encode_result(manuscript::try_compare_reports(&before, &after).map_err(super::reason)?)
}

#[napi(
    js_name = "parseManuscriptReport",
    ts_args_type = "value: unknown",
    ts_return_type = "import('./manuscript').Report"
)]
pub fn parse_manuscript_report(value: serde_json::Value) -> napi::Result<serde_json::Value> {
    let report: Report = super::decode_input(value, "invalid report object")?;
    report.validate().map_err(super::reason)?;
    super::encode_result(report)
}
#[napi(
    js_name = "defaultManuscriptConfiguration",
    ts_return_type = "import('./manuscript').ResolvedConfiguration"
)]
pub fn default_manuscript_configuration() -> napi::Result<serde_json::Value> {
    super::encode_result(manuscript::Configuration::default())
}
#[napi(
    js_name = "listManuscriptRules",
    ts_return_type = "Array<import('./manuscript').RuleDefinition>"
)]
pub fn list_manuscript_rules() -> napi::Result<serde_json::Value> {
    super::encode_result(manuscript::rule_definitions())
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
        self.manifest.validate(self.kind).map_err(super::reason)?;
        if self.directory.trim().is_empty()
            || self.directory.len() > 4096
            || self.directory.contains('\0')
        {
            return Err(super::reason("invalid model directory"));
        }
        #[cfg(feature = "onnx")]
        {
            super::encode_result(
                crate::onnx::set_model_directory_verified(
                    self.kind,
                    &self.directory,
                    &self.manifest,
                )
                .map_err(super::reason)?,
            )
        }
        #[cfg(not(feature = "onnx"))]
        {
            Err(super::reason(
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
    let kind = super::decode_input(serde_json::Value::String(kind), "unsupported model kind")?;
    let manifest = super::decode_input(manifest, "invalid model manifest fields")?;
    if model_root.trim().is_empty() || model_root.len() > 4096 || model_root.contains('\0') {
        return Err(super::reason("invalid model directory"));
    }
    let directory = std::path::PathBuf::from(model_root);
    let directory = if directory.is_absolute() {
        directory
    } else {
        std::env::current_dir()
            .map_err(|_| super::reason("cannot resolve model directory"))?
            .join(directory)
    };
    Ok(AsyncTask::new(LoadVerifiedModelTask {
        kind,
        directory: directory
            .to_str()
            .ok_or_else(|| super::reason("model directory must be valid Unicode"))?
            .into(),
        manifest,
    }))
}
