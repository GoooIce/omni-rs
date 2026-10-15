use crate::engine_harness::{Adapter, Context, Event, Lifecycle};
use omni_hy_mt::{Error, Model, Precision, Request, Translator};
use std::path::{Path, PathBuf};
#[cfg(target_os = "macos")]
type Backend = burn::backend::wgpu::Metal;
#[cfg(not(target_os = "macos"))]
type Backend = burn::backend::wgpu::WebGpu;

pub enum Command {
    Load(PathBuf),
    Translate { job_id: u64, request: Request },
}
#[derive(serde::Serialize)]
pub struct Update {
    pub kind: &'static str,
    pub message: String,
    pub text: String,
    pub job_id: u64,
}
impl Update {
    fn status(kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            text: String::new(),
            job_id: 0,
        }
    }
}
impl Event for Update {
    fn lifecycle(kind: Lifecycle, message: String, _progress: f64) -> Self {
        Self::status(kind.as_str(), message)
    }
    fn trace(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}
pub struct Worker {
    translator: Option<Translator<Backend>>,
}
pub type Engine = crate::engine_harness::Engine<Worker>;
impl Adapter for Worker {
    type Command = Command;
    type Update = Update;
    type Error = Error;
    const NAME: &'static str = "translation";
    fn new() -> Self {
        Self { translator: None }
    }
    fn unload(&mut self) {
        self.translator = None;
    }
    fn load_path(command: &Command) -> Option<&Path> {
        match command {
            Command::Load(path) => Some(path),
            _ => None,
        }
    }
    fn check_files(path: &Path) -> Result<(), String> {
        crate::model_location::check_files(path)
    }
    fn initialize() {
        crate::gpu::initialize();
    }
    fn is_cancelled(error: &Error) -> bool {
        matches!(error, Error::Interrupted)
    }
    fn load(&mut self, path: &Path, context: &Context<Update>) -> omni_hy_mt::Result<Update> {
        let model = Model::from_directory(
            path,
            Precision::F16,
            &Default::default(),
            &context.stop,
            |n, total| {
                context.loading(
                    n as f64 / total as f64,
                    format!("正在加载 Hy-MT2-7B · {n}/{total}"),
                );
            },
        )?;
        self.translator = Some(Translator::new(model, 4096));
        Ok(Update::status("ready", "Hy-MT2-7B 已就绪"))
    }
    fn run(&mut self, command: Command, context: &Context<Update>) -> omni_hy_mt::Result<Update> {
        let Command::Translate { job_id, request } = command else {
            unreachable!("load handled by harness")
        };
        let translator = self
            .translator
            .as_mut()
            .ok_or_else(|| Error::Config("请先载入翻译模型".into()))?;
        let output = pollster::block_on(translator.translate(&request, &context.stop, |event| {
            context.updates.publish(Update {
                kind: "translation",
                job_id,
                text: event.text,
                message: "正在翻译…".into(),
            });
        }))?;
        Ok(Update {
            kind: if output.finish_reason == "cancelled" {
                "cancelled"
            } else {
                "done"
            },
            job_id,
            text: output.text,
            message: format!(
                "{} · {:.2} 秒 · 复用 {} 个前缀 token",
                match output.finish_reason.as_str() {
                    "cancelled" => "已停止翻译",
                    "length" => "已达输出上限，译文未完整结束",
                    _ => "翻译完成",
                },
                output.compute_ms / 1000.0,
                output.cached_tokens
            ),
        })
    }
}
