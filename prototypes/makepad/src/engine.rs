use crate::asr::read_wav;
use crate::engine_harness::Failure;
use crate::engine_harness::{Adapter, Context, Event, Lifecycle};
use crate::gpu::{self, Backend};
use crate::protocol::Update;
use crate::resampler::NativeResampler;
use crate::transcription_start::stream_config;
use omni_hub::providers::{
    encode_mono_wav_16k, map_asr_language_to_iso, ExternalProviderTarget, ProviderError,
    TranscriptionRequest,
};
use omni_r2t2::{Model, Transcription, TranscriptionEvent};
use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

pub enum Command {
    Load(String),
    Transcribe {
        path: String,
        context: String,
        mode: usize,
        language: Option<String>,
    },
    Microphone {
        context: String,
        mode: usize,
        language: Option<String>,
    },
    TranscribeExternal {
        path: String,
        context: String,
        language: Option<String>,
        target: ExternalProviderTarget,
    },
    MicrophoneExternal {
        context: String,
        language: Option<String>,
        target: ExternalProviderTarget,
        finish_recording: Arc<AtomicBool>,
    },
}

pub type Engine = crate::engine_harness::Engine<Worker>;

pub struct Worker {
    model: Option<Arc<Model<Backend>>>,
    device: Option<gpu::Device>,
}

impl Event for Update {
    fn lifecycle(kind: Lifecycle, message: String, progress: f64) -> Self {
        let mut update = Self::status(kind.as_str(), message);
        update.progress = progress;
        update
    }
    fn trace(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

impl Adapter for Worker {
    type Command = Command;
    type Update = Update;
    type Error = Failure;
    fn is_cancelled(error: &Failure) -> bool {
        matches!(error, Failure::Cancelled)
    }
    const NAME: &'static str = "r2t2";

    fn new() -> Self {
        // Defer GPU/ANE device initialization until a local model is loaded so
        // external-only ASR runs never touch local hardware devices.
        Self {
            model: None,
            device: None,
        }
    }
    fn unload(&mut self) {
        self.model = None;
    }
    fn load_path(command: &Command) -> Option<&Path> {
        if let Command::Load(path) = command {
            Some(Path::new(path))
        } else {
            None
        }
    }
    fn check_files(path: &Path) -> Result<(), String> {
        crate::model_location::check_files(path)
    }
    fn after_command(&self) {
        if let Some(device) = &self.device {
            gpu::log_stats(device, "asr");
        }
    }

    fn load(&mut self, path: &Path, context: &Context<Update>) -> Result<Update, Failure> {
        let start = Instant::now();
        let device = self.device.get_or_insert_with(gpu::device);
        self.model = Some(Arc::new(
            crate::model_loader::load(
                path,
                gpu::default_precision(),
                device,
                &context.stop,
                |progress, message| context.loading(progress, message),
            )
            .map_err(Failure::from)?,
        ));
        let mut ready = Update::status(
            "ready",
            format!("模型已就绪 · 载入 {:.1} s", start.elapsed().as_secs_f64()),
        );
        ready.progress = 1.0;
        ready.backend = gpu::backend_label().into();
        Ok(ready)
    }

    fn run(&mut self, command: Command, job: &Context<Update>) -> Result<Update, Failure> {
        let stop = &job.stop;
        match command {
            Command::Load(_) => unreachable!("load is handled by the harness"),
            Command::TranscribeExternal {
                path,
                context,
                language,
                target,
            } => run_external_file(&path, &context, language.as_deref(), &target, job),
            Command::MicrophoneExternal {
                context,
                language,
                target,
                finish_recording,
            } => run_external_microphone(
                &context,
                language.as_deref(),
                &target,
                finish_recording,
                job,
            ),
            Command::Transcribe {
                path,
                context,
                mode,
                language,
            } => {
                let model = self.model.as_ref().ok_or("请先载入模型")?.clone();
                let pcm = read_wav(&path)?;
                let mut runner =
                    Transcription::new(model, stream_config(mode, false, language, context), 16000)
                        .map_err(Failure::from)?;
                runner.set_stop_handle(stop.clone());
                for packet in pcm.chunks(320) {
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    match feed(&mut runner, packet, job) {
                        Ok(()) => (),
                        Err(omni_r2t2::Error::Interrupted) => break,
                        Err(error) => return Err(error.into()),
                    }
                }
                finish(&mut runner, job)
            }
            Command::Microphone {
                context,
                mode,
                language,
            } => {
                let model = self.model.as_ref().ok_or("请先载入模型")?.clone();
                let mut runner =
                    Transcription::new(model, stream_config(mode, true, language, context), 16000)
                        .map_err(Failure::from)?;
                runner.set_stop_handle(stop.clone());
                let microphone =
                    crate::microphone::Microphone::start(stop.clone(), job.updates.clone())?;
                let mut source = crate::audio_spool::Reader::new(microphone.shared.spool.clone());
                let mut resampler =
                    NativeResampler::new(microphone.rate, 16000).map_err(str::to_owned)?;
                let mut packet = vec![0.0; (microphone.rate / 50) as usize];
                loop {
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    let n = source
                        .pop(&mut packet)
                        .map_err(|e| format!("读取录音缓存失败：{e}"))?;
                    if n > 0 {
                        let pcm = resampler.push(&packet[..n], false).map_err(str::to_owned)?;
                        match feed(&mut runner, &pcm, job) {
                            Ok(()) => (),
                            Err(omni_r2t2::Error::Interrupted) => break,
                            Err(error) => return Err(error.into()),
                        }
                    } else if microphone.shared.finished.load(Ordering::Acquire) {
                        // The producer may have committed after the first pop.
                        if microphone.shared.spool.pending() == 0 {
                            break;
                        }
                    } else {
                        microphone
                            .shared
                            .spool
                            .wait_readable(stop, &microphone.shared.finished);
                    }
                }
                if !stop.load(Ordering::Acquire) {
                    let tail = resampler.push(&[], true).map_err(str::to_owned)?;
                    match feed(&mut runner, &tail, job) {
                        Ok(()) | Err(omni_r2t2::Error::Interrupted) => (),
                        Err(error) => return Err(error.into()),
                    }
                }
                let mut final_update = finish(&mut runner, job)?;
                final_update.message = microphone.shared.warning().into();
                if final_update.message.is_empty() {
                    final_update.message = if final_update.cancelled {
                        "已打断 · 保留已确认文字，未处理音频已取消"
                    } else {
                        "录音已结束 · 尾音处理完成"
                    }
                    .into();
                }
                final_update.captured_seconds =
                    microphone.shared.buffer.captured() as f64 / microphone.rate as f64;
                // Stop/join the device before the harness admits another job.
                drop(microphone);
                drop(source);
                Ok(final_update)
            }
        }
    }
}

fn run_external_file(
    path: &str,
    context: &str,
    language: Option<&str>,
    target: &ExternalProviderTarget,
    job: &Context<Update>,
) -> Result<Update, Failure> {
    let pcm = read_wav(path)?;
    let audio_seconds = pcm.len() as f64 / 16_000.0;
    let filename = Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "audio.wav".to_owned());

    upload_external_pcm(
        ExternalUpload {
            pcm: &pcm,
            filename,
            context,
            language,
            target,
            captured_seconds: 0.0,
            progress_message: format!(
                "正在上传整段音频（{audio_seconds:.1} s）至 {} 转写…",
                target.display_badge()
            ),
            done_message: format!("外部整段转写完成 · {}", target.display_badge()),
        },
        job,
    )
}

fn run_external_microphone(
    context: &str,
    language: Option<&str>,
    target: &ExternalProviderTarget,
    finish_recording: Arc<AtomicBool>,
    job: &Context<Update>,
) -> Result<Update, Failure> {
    let capture_stop = Arc::new(AtomicBool::new(false));
    let microphone =
        crate::microphone::Microphone::start(capture_stop.clone(), job.updates.clone())?;
    let mut source = crate::audio_spool::Reader::new(microphone.shared.spool.clone());
    let mut resampler = NativeResampler::new(microphone.rate, 16_000).map_err(str::to_owned)?;
    let mut packet = vec![0.0; (microphone.rate / 50) as usize];
    let mut recorded_pcm: Vec<f32> = Vec::new();

    loop {
        if job.stop.load(Ordering::Acquire) {
            capture_stop.store(true, Ordering::Release);
            drop(microphone);
            drop(source);
            return Err(Failure::Cancelled);
        }
        if finish_recording.load(Ordering::Acquire) {
            capture_stop.store(true, Ordering::Release);
        }
        let n = source
            .pop(&mut packet)
            .map_err(|e| format!("读取录音缓存失败：{e}"))?;
        if n > 0 {
            let chunk = resampler.push(&packet[..n], false).map_err(str::to_owned)?;
            recorded_pcm.extend_from_slice(&chunk);
            if recorded_pcm.len() > 16_000 * 600 {
                capture_stop.store(true, Ordering::Release);
                break;
            }
        } else if microphone.shared.finished.load(Ordering::Acquire) {
            if microphone.shared.spool.pending() == 0 {
                break;
            }
        } else if capture_stop.load(Ordering::Acquire) {
            std::thread::sleep(std::time::Duration::from_millis(10));
        } else {
            std::thread::sleep(std::time::Duration::from_millis(15));
        }
    }

    let captured_seconds = microphone.shared.buffer.captured() as f64 / microphone.rate as f64;
    let warning = microphone.shared.warning().to_owned();
    drop(microphone);
    drop(source);

    if job.stop.load(Ordering::Acquire) {
        return Err(Failure::Cancelled);
    }

    let tail = resampler.push(&[], true).map_err(str::to_owned)?;
    recorded_pcm.extend_from_slice(&tail);
    if recorded_pcm.is_empty() {
        return Err(Failure::Error("录音为空，未上传至外部网关".into()));
    }

    let audio_seconds = recorded_pcm.len() as f64 / 16_000.0;
    let done_message = if warning.is_empty() {
        format!("录音批处理转写完成 · {}", target.display_badge())
    } else {
        format!("{warning} · 外部转写完成")
    };
    upload_external_pcm(
        ExternalUpload {
            pcm: &recorded_pcm,
            filename: "microphone.wav".to_owned(),
            context,
            language,
            target,
            captured_seconds,
            progress_message: format!(
                "录音结束（{audio_seconds:.1} s）· 正在上传至 {} 批处理转写…",
                target.display_badge()
            ),
            done_message,
        },
        job,
    )
}

struct ExternalUpload<'a> {
    pcm: &'a [f32],
    filename: String,
    context: &'a str,
    language: Option<&'a str>,
    target: &'a ExternalProviderTarget,
    captured_seconds: f64,
    progress_message: String,
    done_message: String,
}

fn upload_external_pcm(
    upload: ExternalUpload<'_>,
    job: &Context<Update>,
) -> Result<Update, Failure> {
    let ExternalUpload {
        pcm,
        filename,
        context,
        language,
        target,
        captured_seconds,
        progress_message,
        done_message,
    } = upload;
    let audio_seconds = pcm.len() as f64 / 16_000.0;
    let wav_bytes = encode_mono_wav_16k(pcm).map_err(|e| Failure::Error(e.to_string()))?;

    job.updates
        .publish(Update::status("progress", progress_message));

    let upload_start = Instant::now();
    let client = target.client().map_err(|e| Failure::Error(e.to_string()))?;
    let resp = client
        .transcribe_wav(
            &TranscriptionRequest {
                model: target.model.clone(),
                wav_bytes,
                filename,
                language_iso: language
                    .and_then(map_asr_language_to_iso)
                    .map(str::to_owned),
                prompt: (!context.trim().is_empty()).then(|| context.trim().to_owned()),
            },
            &job.stop,
        )
        .map_err(map_provider_error)?;

    let elapsed_ms = upload_start.elapsed().as_secs_f64() * 1000.0;
    Ok(Update {
        kind: "done".into(),
        text: resp.text,
        message: done_message,
        compute_ms: elapsed_ms,
        total_compute_ms: elapsed_ms,
        audio_seconds,
        captured_seconds,
        rtf: if audio_seconds > 0.0 {
            (elapsed_ms / 1000.0) / audio_seconds
        } else {
            0.0
        },
        event_count: 1,
        language: resp.language.unwrap_or_default(),
        ..Default::default()
    })
}

fn map_provider_error(err: ProviderError) -> Failure {
    match err {
        ProviderError::Cancelled => Failure::Cancelled,
        other => Failure::Error(other.to_string()),
    }
}

fn feed(
    runner: &mut Transcription<Backend>,
    pcm: &[f32],
    job: &Context<Update>,
) -> omni_r2t2::Result<()> {
    for event in pollster::block_on(runner.push(pcm))? {
        job.updates.publish(update(event, "event"));
    }
    Ok(())
}

fn finish(runner: &mut Transcription<Backend>, job: &Context<Update>) -> Result<Update, Failure> {
    let mut events = pollster::block_on(runner.finish()).map_err(Failure::from)?;
    let terminal = events.pop().ok_or("转写任务缺少最终事件")?;
    for event in events {
        job.updates.publish(update(event, "event"));
    }
    Ok(update(terminal, "done"))
}

fn update(report: TranscriptionEvent, kind: &str) -> Update {
    Update {
        kind: kind.into(),
        text: report.event.text,
        delta: report.event.delta,
        compute_ms: report.event.compute_ms,
        total_compute_ms: report.total_compute_ms,
        audio_seconds: report.event.audio_seconds,
        rtf: report.rtf,
        event_count: report.event_count,
        cancelled: report.cancelled,
        language: report.event.language,
        ..Default::default()
    }
}
