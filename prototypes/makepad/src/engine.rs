use crate::asr::{read_wav, stream_config};
use crate::engine_harness::Failure;
use crate::engine_harness::{Adapter, Context, Event, Lifecycle};
use crate::gpu::{self, Backend};
use crate::protocol::{DecodeOptions, Update};
use crate::resampler::NativeResampler;
use omni_r2t2::{Model, Precision, Transcription, TranscriptionEvent};
use std::{
    path::Path,
    sync::{atomic::Ordering, Arc},
    time::Instant,
};

pub enum Command {
    Load(String),
    Transcribe {
        path: String,
        context: String,
        options: DecodeOptions,
    },
    Microphone {
        context: String,
        options: DecodeOptions,
    },
}

pub type Engine = crate::engine_harness::Engine<Worker>;

pub struct Worker {
    model: Option<Arc<Model<Backend>>>,
    device: gpu::Device,
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
        Self {
            model: None,
            device: gpu::device(),
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
    fn initialize() {
        gpu::initialize();
    }
    fn after_command(&self) {
        gpu::log_stats(&self.device, "asr");
    }

    fn load(&mut self, path: &Path, context: &Context<Update>) -> Result<Update, Failure> {
        let start = Instant::now();
        self.model = Some(Arc::new(
            crate::model_loader::load(
                path,
                Precision::F16,
                &self.device,
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
        let model = self.model.as_ref().ok_or("请先载入模型")?.clone();
        let stop = &job.stop;
        match command {
            Command::Load(_) => unreachable!("load is handled by the harness"),
            Command::Transcribe {
                path,
                context,
                options,
            } => {
                let pcm = read_wav(&path)?;
                let mut runner =
                    Transcription::new(model, stream_config(options, context, false), 16000)
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
            Command::Microphone { context, options } => {
                let mut runner =
                    Transcription::new(model, stream_config(options, context, true), 16000)
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
        ..Default::default()
    }
}
