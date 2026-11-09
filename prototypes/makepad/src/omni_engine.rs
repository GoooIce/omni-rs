//! LFM3 conversation worker; the desktop harness owns admission and stopping.
use crate::engine_harness::{Adapter, Context, Event, Lifecycle};
use omni_lfm3::{
    generate::{EndReason, OmniEvent, OmniOptions, OutputMode},
    model::{Conversation, InputPart, OmniModel},
    Error, Precision, Result,
};
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

type Backend = omni_lfm3::NativeGpu;

#[derive(Clone)]
pub enum Part {
    Text(String),
    File(PathBuf),
}
pub enum Command {
    Load(PathBuf),
    Generate {
        parts: Vec<Part>,
        mode: OutputMode,
        audio_temperature: Option<f32>,
        audio_top_k: Option<usize>,
    },
    Listen(Arc<AtomicBool>),
    ListenStable(Arc<AtomicBool>),
    Clear,
    Replay(Arc<[f32]>),
}
#[derive(serde::Serialize)]
pub struct Update {
    pub kind: &'static str,
    pub message: String,
    pub text: String,
    #[serde(skip)]
    pub audio: Option<Arc<[f32]>>,
    pub can_continue: bool,
}
impl Update {
    fn status(kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            text: String::new(),
            audio: None,
            can_continue: true,
        }
    }
}
impl Event for Update {
    fn lifecycle(kind: Lifecycle, message: String, _: f64) -> Self {
        Self::status(kind.as_str(), message)
    }
    fn trace(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}
pub type Engine = crate::engine_harness::Engine<Worker>;
pub struct Worker {
    model: Option<OmniModel<Backend>>,
    conversation: Option<Conversation<Backend>>,
}
impl Adapter for Worker {
    type Command = Command;
    type Update = Update;
    type Error = Error;
    const NAME: &'static str = "lfm3-omni";
    fn new() -> Self {
        Self {
            model: None,
            conversation: None,
        }
    }
    fn unload(&mut self) {
        self.conversation = None;
        self.model = None;
    }
    fn load_path(command: &Command) -> Option<&Path> {
        match command {
            Command::Load(path) => Some(path),
            _ => None,
        }
    }
    fn check_files(path: &Path) -> std::result::Result<(), String> {
        crate::model_location::require_files(
            path,
            &[
                "config.json",
                "processor_config.json",
                "generation_config.json",
                "chat_template.jinja",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        )?;
        crate::model_location::check_weights(path, "model")?;
        let codec = path.join("audio_tokenizer");
        crate::model_location::require_files(&codec, &["config.json"])?;
        crate::model_location::check_weights(&codec, "model")
    }
    fn is_cancelled(error: &Error) -> bool {
        matches!(error, Error::Interrupted)
    }
    fn uses_model(command: &Command) -> bool {
        !matches!(command, Command::Replay(_))
    }
    fn load(&mut self, path: &Path, context: &Context<Update>) -> Result<Update> {
        context.loading(0.0, "正在载入 LFM3-Omni-2B…");
        let model = OmniModel::<Backend>::load(path, Precision::F16, &Default::default())?;
        self.conversation = Some(Conversation::new(&model));
        self.model = Some(model);
        Ok(Update::status("ready", "LFM3-Omni-2B 已就绪"))
    }
    fn run(&mut self, command: Command, context: &Context<Update>) -> Result<Update> {
        match command {
            Command::Load(_) => unreachable!("load is handled by the harness"),
            Command::Clear => {
                self.conversation
                    .as_mut()
                    .ok_or_else(|| Error::Config("请先载入模型".into()))?
                    .clear();
                Ok(Update::status("cleared", "已开始新对话"))
            }
            Command::Replay(audio) => {
                context
                    .updates
                    .publish(Update::status("playing", "正在播放语音…"));
                let mut playback = crate::playback::Playback::open(context.stop.clone(), None)
                    .map_err(Error::Audio)?;
                playback.push(&audio).map_err(Error::Audio)?;
                playback.drain().map_err(Error::Audio)?;
                Ok(Update::status(
                    "replayed",
                    if context.stop.load(Ordering::Acquire) {
                        "播放已停止"
                    } else {
                        "播放结束"
                    },
                ))
            }
            Command::Listen(finish) => {
                let model = self
                    .model
                    .as_ref()
                    .ok_or_else(|| Error::Config("请先载入模型".into()))?;
                listen(model, finish, context)
            }
            Command::ListenStable(finish) => {
                let model = self
                    .model
                    .as_ref()
                    .ok_or_else(|| Error::Config("请先载入模型".into()))?;
                listen_stable(model, finish, context)
            }
            Command::Generate {
                parts,
                mode,
                audio_temperature,
                audio_top_k,
            } => {
                let model = self
                    .model
                    .as_ref()
                    .ok_or_else(|| Error::Config("请先载入模型".into()))?;
                let conversation = self
                    .conversation
                    .as_mut()
                    .ok_or_else(|| Error::Config("对话尚未初始化".into()))?;
                let parts = parts
                    .into_iter()
                    .map(read_part)
                    .collect::<Result<Vec<_>>>()?;
                context
                    .updates
                    .publish(Update::status("preparing", "正在处理输入…"));
                let (message, media) = model.prepare_parts(&parts, &context.stop)?;
                let options = OmniOptions {
                    mode,
                    max_slots: 1500,
                    audio_temperature,
                    audio_top_k,
                    ..Default::default()
                };
                let mut tokens = Vec::new();
                let mut error = None;
                let result = conversation.generate(
                    model,
                    &[message],
                    None,
                    &media,
                    &options,
                    &context.stop,
                    |event| {
                        if let OmniEvent::TextToken(token) = event {
                            tokens.push(*token);
                            match model.tokenizer.decode(&tokens, true) {
                                Ok(text) => context.updates.publish(Update {
                                    text,
                                    ..Update::status("text", "正在生成…")
                                }),
                                Err(e) => {
                                    error = Some(e);
                                    return false;
                                }
                            }
                        } else if matches!(event, OmniEvent::SpeechCodes(_)) {
                            context
                                .updates
                                .try_publish(Update::status("speech", "正在生成语音…"));
                        }
                        !context.stop.load(Ordering::Acquire)
                    },
                )?;
                if let Some(error) = error {
                    return Err(error);
                }
                let (kind, message) = match result.reason {
                    EndReason::Stop(_) => ("done", "回答完成"),
                    EndReason::Limit => ("done", "已达到输出上限；请开始新对话"),
                    EndReason::Cancelled => ("cancelled", "已停止；已交付内容保留，请开始新对话"),
                };
                Ok(Update {
                    kind,
                    message: message.into(),
                    text: model.tokenizer.decode(&tokens, true)?,
                    audio: (!result.samples.is_empty()).then(|| Arc::from(result.samples)),
                    can_continue: conversation.can_continue(),
                })
            }
        }
    }
}

fn listen(
    model: &OmniModel<Backend>,
    finish: Arc<AtomicBool>,
    context: &Context<Update>,
) -> Result<Update> {
    use omni_lfm3::live::LiveTranscription;
    let capture_stop = Arc::new(AtomicBool::new(false));
    let updates = crate::engine_harness::Publisher::<crate::protocol::Update>::discarding(
        capture_stop.clone(),
    );
    let microphone = crate::microphone::Microphone::start(capture_stop.clone(), updates)
        .map_err(Error::Audio)?;
    let mut source = crate::audio_spool::Reader::new(microphone.shared.spool.clone());
    let mut resampler = crate::resampler::NativeResampler::new(microphone.rate, 16_000)
        .map_err(|e| Error::Audio(e.into()))?;
    let mut live = LiveTranscription::new(model, &model.live_transcription_prompt()?)?;
    let mut tokens = Vec::new();
    let mut pending = Vec::new();
    let mut packet = vec![0.; (microphone.rate / 50) as usize];
    let label = if std::env::args().any(|arg| arg == "--mic-test-wav") {
        "测试音源（非麦克风）"
    } else {
        "录音中"
    };
    context.updates.publish(Update::status(
        "listening",
        format!("{label} · 实时文字仅供预览，结束后校准中文全文"),
    ));
    loop {
        if context.stop.load(Ordering::Acquire) {
            capture_stop.store(true, Ordering::Release);
            break;
        }
        if finish.load(Ordering::Acquire) {
            capture_stop.store(true, Ordering::Release);
        }
        let count = source
            .pop(&mut packet)
            .map_err(|error| Error::Audio(format!("读取录音缓存失败：{error}")))?;
        if count > 0 {
            pending.extend(
                resampler
                    .push(&packet[..count], false)
                    .map_err(|e| Error::Audio(e.into()))?,
            );
            while pending.len() >= 16_000 {
                let chunk: Vec<_> = pending.drain(..16_000).collect();
                feed_live(model, &mut live, &chunk, false, context, &mut tokens)?;
                if context.stop.load(Ordering::Acquire) {
                    break;
                }
            }
        } else if microphone.shared.finished.load(Ordering::Acquire)
            && microphone.shared.spool.pending() == 0
        {
            break;
        } else {
            microphone
                .shared
                .spool
                .wait_readable(&context.stop, &microphone.shared.finished);
        }
    }
    if !context.stop.load(Ordering::Acquire) {
        pending.extend(
            resampler
                .push(&[], true)
                .map_err(|e| Error::Audio(e.into()))?,
        );
        feed_live(model, &mut live, &pending, true, context, &mut tokens)?;
    }
    let warning = microphone.shared.warning().to_owned();
    drop(microphone);
    drop(source);
    let mut limited = false;
    if !context.stop.load(Ordering::Acquire) {
        context
            .updates
            .publish(Update::status("refining", "正在校准完整录音…"));
        let mut refined = Vec::new();
        let mut decoding_error = None;
        let result =
            model.transcribe_audio(live.into_audio()?, "Chinese", 256, &context.stop, |token| {
                refined.push(token);
                match model.tokenizer.decode(&refined, true) {
                    Ok(text) => context.updates.publish(Update {
                        text,
                        ..Update::status("heard_final", "正在校准完整录音…")
                    }),
                    Err(error) => {
                        decoding_error = Some(error);
                        return false;
                    }
                }
                !context.stop.load(Ordering::Acquire)
            })?;
        if let Some(error) = decoding_error {
            return Err(error);
        }
        limited = result.reason == EndReason::Limit;
        if !refined.is_empty() {
            tokens = refined;
        }
    }
    let stopped = context.stop.load(Ordering::Acquire);
    let message = if !warning.is_empty() {
        warning
    } else if stopped {
        "已停止听写，保留已识别文字".into()
    } else if limited {
        "完整转写已达长度上限，文字可能不完整".into()
    } else {
        "听写结束 · 文字已放入输入框".into()
    };
    Ok(Update {
        kind: if stopped {
            "heard_cancelled"
        } else {
            "heard_done"
        },
        message,
        text: model.tokenizer.decode(&tokens, true)?,
        audio: None,
        can_continue: true,
    })
}
fn listen_stable(
    model: &OmniModel<Backend>,
    finish: Arc<AtomicBool>,
    context: &Context<Update>,
) -> Result<Update> {
    use omni_lfm3::stable::StableTranscription;
    let capture_stop = Arc::new(AtomicBool::new(false));
    let updates = crate::engine_harness::Publisher::<crate::protocol::Update>::discarding(
        capture_stop.clone(),
    );
    let microphone = crate::microphone::Microphone::start(capture_stop.clone(), updates)
        .map_err(Error::Audio)?;
    let mut source = crate::audio_spool::Reader::new(microphone.shared.spool.clone());
    let mut resampler = crate::resampler::NativeResampler::new(microphone.rate, 16_000)
        .map_err(|e| Error::Audio(e.into()))?;
    let mut stream = StableTranscription::new(model, "Chinese", 256)?;
    let mut packet = vec![0.; (microphone.rate / 50) as usize];
    let label = if std::env::args().any(|arg| arg == "--mic-test-wav") {
        "测试音源（非麦克风）"
    } else {
        "录音中"
    };
    context.updates.publish(Update::status(
        "listening",
        format!("{label} · 稳定出字，已确认文字不回退"),
    ));
    let mut reason = None;
    let mut conflict = false;
    loop {
        if context.stop.load(Ordering::Acquire) {
            capture_stop.store(true, Ordering::Release);
            break;
        }
        if finish.load(Ordering::Acquire) {
            capture_stop.store(true, Ordering::Release);
        }
        let count = source
            .pop(&mut packet)
            .map_err(|error| Error::Audio(format!("读取录音缓存失败：{error}")))?;
        if count > 0 {
            let samples = resampler
                .push(&packet[..count], false)
                .map_err(|e| Error::Audio(e.into()))?;
            let output =
                stream.push_pcm(model, &samples, 16_000, false, &context.stop, |event| {
                    context.updates.publish(Update {
                        text: event.text.clone(),
                        ..Update::status("heard_stable", "稳定听写中…")
                    });
                    !context.stop.load(Ordering::Acquire)
                })?;
            if output.reason == Some(EndReason::Cancelled) {
                reason = output.reason;
                capture_stop.store(true, Ordering::Release);
                break;
            }
        } else if microphone.shared.finished.load(Ordering::Acquire)
            && microphone.shared.spool.pending() == 0
        {
            break;
        } else {
            microphone
                .shared
                .spool
                .wait_readable(&context.stop, &microphone.shared.finished);
        }
    }
    if !context.stop.load(Ordering::Acquire) && reason != Some(EndReason::Cancelled) {
        let tail = resampler
            .push(&[], true)
            .map_err(|e| Error::Audio(e.into()))?;
        let output = stream.push_pcm(model, &tail, 16_000, true, &context.stop, |event| {
            context.updates.publish(Update {
                text: event.text.clone(),
                ..Update::status("heard_stable", "稳定听写中…")
            });
            !context.stop.load(Ordering::Acquire)
        })?;
        reason = output.reason;
        conflict = output.conflict;
    }
    let warning = microphone.shared.warning().to_owned();
    drop(microphone);
    drop(source);
    let stopped = context.stop.load(Ordering::Acquire) || reason == Some(EndReason::Cancelled);
    let message = if !warning.is_empty() {
        warning
    } else if stopped {
        "已停止听写，保留已确认文字".into()
    } else if reason == Some(EndReason::Limit) {
        "稳定听写已达长度上限，文字可能不完整".into()
    } else if conflict {
        "稳定听写结束 · 已保留先前确认文字，可能有识别偏差".into()
    } else {
        "稳定听写结束 · 文字已放入输入框".into()
    };
    Ok(Update {
        kind: if stopped {
            "heard_cancelled"
        } else {
            "heard_done"
        },
        message,
        text: stream.text().to_owned(),
        audio: None,
        can_continue: true,
    })
}

fn feed_live(
    model: &OmniModel<Backend>,
    live: &mut omni_lfm3::live::LiveTranscription<Backend>,
    samples: &[f32],
    is_final: bool,
    context: &Context<Update>,
    tokens: &mut Vec<u32>,
) -> Result<()> {
    let mut error = None;
    let result = live.push_pcm(
        model,
        samples,
        16_000,
        is_final,
        &Default::default(),
        &context.stop,
        |token| {
            tokens.push(token);
            match model.tokenizer.decode(tokens, true) {
                Ok(text) => context.updates.publish(Update {
                    text,
                    ..Update::status("heard", "实时听写中…")
                }),
                Err(e) => {
                    error = Some(e);
                    return false;
                }
            }
            !context.stop.load(Ordering::Acquire)
        },
    );
    if let Some(error) = error {
        return Err(error);
    }
    match result {
        Ok(_) | Err(Error::Interrupted) if context.stop.load(Ordering::Acquire) => Ok(()),
        Ok(_) => Ok(()),
        Err(error) => Err(error),
    }
}
fn read_part(part: Part) -> Result<InputPart> {
    match part {
        Part::Text(text) => Ok(InputPart::Text(text)),
        Part::File(path) => {
            let metadata = std::fs::metadata(&path)?;
            if metadata.len() > 64 * 1024 * 1024 {
                return Err(Error::Audio("媒体文件超过 64 MB".into()));
            }
            let bytes = std::fs::read(&path)?;
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            match ext.as_str() {
                "png" | "jpg" | "jpeg" | "webp" => Ok(InputPart::Image(bytes)),
                "wav" | "mp3" | "m4a" | "flac" | "ogg" => Ok(InputPart::Audio(bytes)),
                "mp4" | "mov" | "mkv" | "webm" => Ok(InputPart::Video(bytes)),
                _ => Err(Error::Audio(format!(
                    "不支持的媒体格式：{}",
                    path.display()
                ))),
            }
        }
    }
}
