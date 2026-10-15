//! Native music uses the same cancellable admission as speech and image jobs.
//! Pipeline models are scoped to each request and released before completion.
use crate::engine_harness::{Adapter, Context, Event, Lifecycle};
use omni_music::{pipeline, Error, Precision, Progress, SongRequest};
use std::{
    path::{Path, PathBuf},
    sync::{atomic::Ordering, Arc},
};

#[cfg(target_os = "macos")]
type Backend = burn::backend::wgpu::Metal;
#[cfg(not(target_os = "macos"))]
type Backend = burn::backend::wgpu::WebGpu;

pub enum Job {
    Analyze(PathBuf),
    Transcribe { path: PathBuf, melody_only: bool },
    Generate(SongRequest),
}
pub enum Command {
    Run { root: PathBuf, job: Box<Job> },
    Replay(Arc<[f32]>),
}
pub struct Song {
    pub audio: Arc<[f32]>,
    pub semantic: omni_music::yue::Semantic,
    pub elapsed_seconds: f64,
}
pub enum Output {
    Analysis(omni_music::mert::Analysis),
    Score(omni_music::score::Score),
    Song(Song),
}
#[derive(serde::Serialize)]
pub struct Update {
    pub kind: &'static str,
    pub message: String,
    pub progress: f64,
    #[serde(skip)]
    pub output: Option<Output>,
}
impl Update {
    fn status(kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            progress: 0.0,
            output: None,
        }
    }
}
impl Event for Update {
    fn lifecycle(kind: Lifecycle, message: String, progress: f64) -> Self {
        Self {
            progress,
            ..Self::status(kind.as_str(), message)
        }
    }
    fn trace(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}
pub type Engine = crate::engine_harness::Engine<Worker>;
pub struct Worker;
impl Adapter for Worker {
    type Command = Command;
    type Update = Update;
    type Error = Error;
    const NAME: &'static str = "music";
    fn new() -> Self {
        Self
    }
    fn unload(&mut self) {}
    fn load_path(_: &Command) -> Option<&Path> {
        None
    }
    fn check_files(_: &Path) -> Result<(), String> {
        Ok(())
    }
    fn initialize() {
        crate::gpu::initialize();
    }
    fn load(&mut self, _: &Path, _: &Context<Update>) -> omni_music::Result<Update> {
        unreachable!("music loads the models needed by each operation")
    }
    fn is_cancelled(error: &Error) -> bool {
        matches!(error, Error::Interrupted)
    }
    fn uses_model(command: &Command) -> bool {
        matches!(command, Command::Run { .. })
    }
    fn run(&mut self, command: Command, context: &Context<Update>) -> omni_music::Result<Update> {
        if context.stop.load(Ordering::Acquire) {
            return Err(Error::Interrupted);
        }
        let (root, job) = match command {
            Command::Run { root, job } => (root, job),
            Command::Replay(audio) => {
                context
                    .updates
                    .publish(Update::status("playing", "正在试听 · 48 kHz 双声道"));
                crate::clip_playback::play(audio, context.stop.clone(), |seconds, total| {
                    let mut update = Update::status(
                        "playing",
                        format!("正在试听 · {seconds:.1} / {total:.1} 秒"),
                    );
                    update.progress = seconds / total;
                    context.updates.try_publish(update);
                })
                .map_err(Error::Audio)?;
                if context.stop.load(Ordering::Acquire) {
                    return Err(Error::Interrupted);
                }
                return Ok(Update::status("replayed", "试听结束"));
            }
        };
        let request = match *job {
            Job::Analyze(path) => pipeline::Request::Analyze(read_audio(&path, 360)?),
            Job::Transcribe { path, melody_only } => pipeline::Request::Transcribe {
                audio: read_audio(&path, 300)?,
                melody_only,
            },
            Job::Generate(request) => {
                request.validate()?;
                pipeline::Request::Generate(request)
            }
        };
        check_models(&root, &request).map_err(Error::Config)?;
        Self::initialize();
        let result = pipeline::run::<Backend>(
            &root,
            Precision::F16,
            &Default::default(),
            request,
            &context.stop,
            |p: Progress| {
                let label = match p.stage {
                    "load_mert" => "加载音乐编码器",
                    "mert" => "分析音乐",
                    "load_sheetsage" => "加载转谱模型",
                    "score_tokens" => "生成音乐事件",
                    "load_yue" => "加载歌曲模型",
                    "score_plan" => "创作乐谱",
                    "music_tokens" => "生成歌曲",
                    "music_acoustics" => "合成音乐",
                    "load_music_vae" => "加载音频解码器",
                    "music_decode" => "解码双声道音频",
                    _ => p.stage,
                };
                let mut update =
                    Update::status("progress", format!("{label} · {} / {}", p.current, p.total));
                update.progress = p.current as f64 / p.total.max(1) as f64;
                context.updates.publish(update);
            },
        )?;
        if context.stop.load(Ordering::Acquire) {
            return Err(Error::Interrupted);
        }
        let (message, output) = match result {
            pipeline::Product::Analysis(a) => (
                format!(
                    "特征提取完成 · {:.2} 秒 · {} 帧 × {} 维",
                    a.audio_seconds, a.frames, a.dimensions
                ),
                Output::Analysis(a),
            ),
            pipeline::Product::Score(s) => (
                format!(
                    "转谱完成 · {} 个音乐事件{}",
                    s.events.len(),
                    if s.truncated {
                        " · 达到长度上限"
                    } else {
                        ""
                    }
                ),
                Output::Score(s),
            ),
            pipeline::Product::Song {
                audio,
                semantic,
                elapsed_seconds,
            } => {
                let capped = semantic.truncated || semantic.plan.truncated;
                (
                    format!(
                        "生成完成 · {:.2} 秒 · 48 kHz 双声道{}",
                        audio.len() as f64 / 96000.0,
                        if capped {
                            " · 已达上限，片段可能不完整"
                        } else {
                            ""
                        }
                    ),
                    Output::Song(Song {
                        audio: audio.into(),
                        semantic,
                        elapsed_seconds,
                    }),
                )
            }
        };
        Ok(Update {
            kind: "done",
            message,
            progress: 1.0,
            output: Some(output),
        })
    }
}

fn read_audio(path: &Path, limit: usize) -> omni_music::Result<Vec<f32>> {
    omni_music::audio::read_wav(std::fs::File::open(path)?, limit)
}
fn check_models(root: &Path, request: &pipeline::Request) -> Result<(), String> {
    let names: &[&str] = match request {
        pipeline::Request::Analyze(_) => &["MERT-v2-FullSong"],
        pipeline::Request::Transcribe { .. } => &["MERT-v2-FullSong", "SheetSage2"],
        pipeline::Request::Generate(_) => &["YuE2-3B", "YuE2-Vae"],
    };
    for name in names {
        let dir = root.join(name);
        crate::model_location::require_files(&dir, &["config.json"])?;
        crate::model_location::check_weights(&dir, "model")?;
        if *name == "YuE2-3B" {
            crate::model_location::require_files(&dir, &["qwen.tiktoken"])?;
        }
    }
    Ok(())
}
