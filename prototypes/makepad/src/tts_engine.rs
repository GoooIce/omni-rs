use crate::engine_harness::Failure;
use crate::{
    gpu::{self, Backend},
    model_location,
    playback::{Playback, PlaybackStats},
    playback_buffer::DurationHistory,
};
use omni_breeze_tts::{
    enhancement::{self, Consideration, Source},
    reference::{self, ReferenceEncoder},
    Model, Precision, ReferenceAudio, SynthesisOptions, SAMPLE_RATE,
};
use std::{
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::{atomic::Ordering, Arc},
    time::Instant,
};

fn duration_voice(options: &SynthesisOptions) -> u64 {
    let mut key = std::collections::hash_map::DefaultHasher::new();
    options.instruction.trim().hash(&mut key);
    options.guidance_scale.to_bits().hash(&mut key);
    if let Some(reference) = &options.reference {
        reference.codes.hash(&mut key);
    } else {
        // Without a reference, a new sampling seed creates a different voice
        // candidate. Its speaking rate cannot inherit the previous candidate's.
        options.sampling.seed.hash(&mut key);
    }
    let cjk = options.text.chars().any(|c| matches!(c as u32, 0x3040..=0x30ff | 0x3400..=0x9fff | 0xac00..=0xd7af | 0x20000..=0x313ff));
    let latin = options.text.chars().any(|c| c.is_ascii_alphabetic());
    (cjk, latin).hash(&mut key);
    key.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn designed_candidates_do_not_share_duration_history() {
        let mut options = SynthesisOptions::default();
        options.text = "你好，今天天气很好。让我们一起听听这个声音。".into();
        options.instruction = "A clear, gentle voice.".into();
        let first = duration_voice(&options);
        let mut history = DurationHistory::default();
        for _ in 0..3 {
            history.observe(first, &options.text, 4.0);
        }
        assert!((history.estimate(first, &options.text) - 4.8).abs() < 1e-6);
        options.sampling.seed += 1;
        let second = duration_voice(&options);
        assert_ne!(first, second);
        assert_eq!(
            history.estimate(second, &options.text),
            crate::playback_buffer::estimated_seconds(&options.text)
        );
        // Reusing a reference fixes the voice independently of the sampler.
        options.reference = Some(ReferenceAudio {
            text: "你好".into(),
            codes: vec![[0; 16]],
        });
        let reference = duration_voice(&options);
        options.sampling.seed += 1;
        assert_eq!(reference, duration_voice(&options));
    }
}

pub enum Command {
    Load(PathBuf),
    Synthesize {
        options: SynthesisOptions,
        reference: Option<ReferenceInput>,
        design: bool,
        /// Capture audio without playing it. Used for a blind trio.
        hold: bool,
    },
    Enhance {
        source: Source,
        prompt: String,
        seeds: [u64; 3],
        checkpoint: PathBuf,
        prefill: enhancement::Prefill,
    },
    Draft {
        request: enhancement::DraftRequest,
        seed: u64,
        checkpoint: PathBuf,
        prefill: enhancement::Prefill,
    },
    Remember {
        audio: Arc<[f32]>,
        text: String,
        seed: u64,
    },
    Replay(Arc<[f32]>),
    UseDesign,
}
#[derive(Clone)]
pub enum ReferenceInput {
    File { path: PathBuf, text: String },
    Designed,
}
struct Design {
    text: String,
    seed: u64,
}
#[derive(serde::Serialize)]
pub struct Update {
    pub kind: &'static str,
    pub message: String,
    pub progress: f64,
    pub audio_seconds: f64,
    pub elapsed: f64,
    pub first_audio: Option<f64>,
    pub reusable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub design_seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub playback: Option<PlaybackStats>,
    #[serde(skip)]
    pub audio: Option<Arc<[f32]>>,
    #[serde(skip)]
    pub consideration: Option<Consideration>,
    #[serde(skip)]
    pub draft_text: Option<String>,
}
impl Update {
    fn status(kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            progress: 0.0,
            audio_seconds: 0.0,
            elapsed: 0.0,
            first_audio: None,
            reusable: false,
            design_seed: None,
            reference_text: None,
            playback: None,
            audio: None,
            consideration: None,
            draft_text: None,
        }
    }
}
impl crate::engine_harness::Event for Update {
    fn lifecycle(kind: crate::engine_harness::Lifecycle, message: String, progress: f64) -> Self {
        let mut update = Self::status(kind.as_str(), message);
        update.progress = progress;
        update
    }
    fn trace(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}
fn offset_playback(stats: Option<PlaybackStats>, offset: f64) -> Option<PlaybackStats> {
    stats.map(|mut stats| {
        stats.first_playback_seconds = stats.first_playback_seconds.map(|t| t + offset);
        stats
    })
}

pub fn check_files(directory: &Path) -> Result<(), String> {
    model_location::check_files(directory)?;
    model_location::require_files(
        directory,
        &[
            "audio_tokenizer/config.json",
            "audio_tokenizer/model.safetensors",
        ],
    )
}

pub type Engine = crate::engine_harness::Engine<Worker>;

pub struct Worker {
    device: gpu::Device,
    model: Option<Model<Backend>>,
    model_path: PathBuf,
    reference_encoder: Option<ReferenceEncoder<Backend>>,
    reference_cache: Option<(Vec<f32>, Vec<[u32; 16]>)>,
    pcm: Vec<f32>,
    completed_design: Option<Design>,
    designed_reference: Option<ReferenceAudio>,
    duration_history: DurationHistory,
    enhancer: Option<crate::speech_enhancer::SpeechEnhancer>,
}

impl crate::engine_harness::Adapter for Worker {
    type Command = Command;
    type Update = Update;
    type Error = Failure;
    fn is_cancelled(error: &Failure) -> bool {
        matches!(error, Failure::Cancelled)
    }
    const NAME: &'static str = "breeze";

    fn uses_model(command: &Command) -> bool {
        !matches!(command, Command::Replay(_))
    }
    fn new() -> Self {
        Self {
            device: gpu::device(),
            model: None,
            model_path: PathBuf::new(),
            reference_encoder: None,
            reference_cache: None,
            pcm: Vec::new(),
            completed_design: None,
            designed_reference: None,
            duration_history: DurationHistory::default(),
            enhancer: None,
        }
    }
    fn unload(&mut self) {
        self.model = None;
        self.reference_encoder = None;
        self.reference_cache = None;
        self.designed_reference = None;
        self.duration_history = DurationHistory::default();
        self.enhancer = None;
    }
    fn load_path(command: &Command) -> Option<&Path> {
        if let Command::Load(path) = command {
            Some(path)
        } else {
            None
        }
    }
    fn check_files(path: &Path) -> Result<(), String> {
        check_files(path)
    }
    fn after_command(&self) {
        gpu::log_stats(&self.device, "tts");
    }

    fn load(
        &mut self,
        path: &Path,
        context: &crate::engine_harness::Context<Update>,
    ) -> Result<Update, Failure> {
        self.pcm.clear();
        self.completed_design = None;
        let Self {
            device,
            model,
            model_path,
            ..
        } = self;
        *model_path = path.to_owned();
        let start = Instant::now();
        *model = Some(
            Model::load_cancellable(
                &path,
                Precision::F16,
                device,
                &context.stop,
                |progress, message| {
                    context.loading(progress, message);
                },
            )
            .map_err(Failure::from)?,
        );
        let mut ready = Update::status(
            "ready",
            format!(
                "Breeze-TTS-2 已就绪 · {} · 载入 {:.1} s",
                gpu::backend_label(),
                start.elapsed().as_secs_f64()
            ),
        );
        ready.progress = 1.0;
        Ok(ready)
    }

    fn run(
        &mut self,
        command: Command,
        context: &crate::engine_harness::Context<Update>,
    ) -> Result<Update, Failure> {
        let Self {
            device,
            model,
            model_path,
            reference_encoder,
            reference_cache,
            pcm,
            completed_design,
            designed_reference,
            duration_history,
            enhancer,
        } = self;
        let stop = &context.stop;
        let sender = &context.updates;
        match command {
            Command::Load(_) => unreachable!("load is handled by the harness"),
            Command::Enhance {
                source,
                prompt,
                seeds,
                checkpoint,
                prefill,
            } => {
                enhancement::checkpoint_available(&checkpoint)?;
                if enhancer.is_none() {
                    sender.publish(Update::status("loading", "正在载入 35B 增强模型…"));
                    *enhancer = Some(
                        crate::speech_enhancer::SpeechEnhancer::load(&checkpoint)
                            .map_err(Failure::from)?,
                    );
                }
                let model = model.as_ref().ok_or("语音合成模型尚未就绪")?;
                let samples = crate::speech_enhancer::sample_three(
                    enhancer.as_mut().unwrap(),
                    &prompt,
                    &source,
                    seeds,
                    prefill,
                    stop,
                )?;
                let consideration =
                    enhancement::consider(&prompt, &source, samples, |speech, instruction| {
                        model.prompt_ids(speech, instruction).is_ok()
                    });
                let mut update = Update::status(
                    "enhanced",
                    match &consideration {
                        Consideration::Speak(_) => "增强结果一致，开始合成".into(),
                        Consideration::Choose { .. } => "请把甲、乙、丙都听完再选择".into(),
                        Consideration::Failed(reason) => reason.clone(),
                    },
                );
                update.consideration = Some(consideration);
                Ok(update)
            }
            Command::Draft {
                request,
                seed,
                checkpoint,
                prefill,
            } => {
                enhancement::checkpoint_available(&checkpoint)?;
                if enhancer.is_none() {
                    sender.publish(Update::status("loading", "正在载入 35B 增强模型…"));
                    *enhancer = Some(
                        crate::speech_enhancer::SpeechEnhancer::load(&checkpoint)
                            .map_err(Failure::from)?,
                    );
                }
                let text = enhancer.as_mut().unwrap().complete(
                    &request.system,
                    &request.user,
                    seed,
                    prefill,
                    stop,
                )?;
                if text.trim().is_empty() {
                    return Err("起草没有得到增强提示".into());
                }
                let mut update = Update::status("drafted", "增强提示草稿已显示，可修改后接受");
                update.draft_text = Some(text);
                Ok(update)
            }
            Command::Remember { audio, text, seed } => {
                *pcm = audio.to_vec();
                *completed_design = Some(Design { text, seed });
                let mut update = Update::status("remembered", "已选定这一版音色");
                update.reusable = true;
                update.design_seed = Some(seed);
                Ok(update)
            }
            Command::Synthesize {
                mut options,
                reference,
                design,
                hold,
            } => {
                let model = model.as_ref().ok_or("语音合成模型尚未就绪")?;
                pcm.clear();
                *completed_design = None;
                if design
                    && (options.instruction.trim().is_empty()
                        || reference.is_some()
                        || options.reference.is_some())
                {
                    return Err("音色设计需要声音描述，请使用无参考音频的设计模式".into());
                }
                let start = Instant::now();
                if matches!(&reference, Some(ReferenceInput::Designed)) {
                    options.reference =
                        Some(designed_reference.clone().ok_or("请先设计并选用一个音色")?);
                    sender.publish(Update::status(
                        "reference_cached",
                        "使用已设计音色 · 正在生成首段音频",
                    ));
                }
                if let Some(ReferenceInput::File { path, text }) = reference {
                    sender.publish(Update::status("reference", "读取参考 WAV…"));
                    let samples = reference::read_wav(&path).map_err(Failure::from)?;
                    let codes = if let Some((cached, codes)) = reference_cache.as_ref() {
                        (cached == &samples).then(|| codes.clone())
                    } else {
                        None
                    };
                    let codes = if let Some(codes) = codes {
                        sender.publish(Update::status(
                            "reference_cached",
                            "参考声音已缓存 · 正在生成首段音频",
                        ));
                        Some(codes)
                    } else if stop.load(Ordering::Acquire) {
                        None
                    } else {
                        if reference_encoder.is_none() {
                            sender.publish(Update::status("reference", "首次加载参考音频编码器…"));
                            *reference_encoder = Some(
                                ReferenceEncoder::load_cancellable(
                                    &model_path.join("audio_tokenizer"),
                                    device,
                                    &stop,
                                )
                                .map_err(Failure::from)?,
                            );
                        }
                        let encoded = reference_encoder
                            .as_ref()
                            .unwrap()
                            .encode(&samples, &stop, |p| {
                                let mut update = Update::status(
                                    "reference",
                                    format!("正在编码参考声音 · {:.0}%", p * 100.0),
                                );
                                update.progress = p;
                                sender.publish(update);
                            })
                            .map_err(Failure::from)?;
                        encoded.map(|encoded| {
                            *reference_cache = Some((samples, encoded.codes.clone()));
                            encoded.codes
                        })
                    };
                    if let Some(codes) = codes {
                        options.reference = Some(ReferenceAudio { text, codes });
                    } else {
                        return Ok(Update::status("done", "已停止 · 尚未生成音频"));
                    }
                }
                // Keep synthesis/save usable on machines with no audio output.
                let playback_offset = start.elapsed().as_secs_f64();
                let voice = duration_voice(&options);
                let mut playback = if hold {
                    None
                } else {
                    match Playback::open(
                        stop.clone(),
                        Some(
                            duration_history
                                .estimate(voice, &options.text)
                                .min(options.max_frames as f64 * 0.08),
                        ),
                    ) {
                        Ok(output) => Some(output),
                        Err(error) => {
                            sender.publish(Update::status(
                                "notice",
                                format!("{error} · 仍可合成并保存 WAV"),
                            ));
                            None
                        }
                    }
                };
                let mut first = None;
                let report = model
                    .synthesize(&options, &stop, |audio| {
                        if stop.load(Ordering::Acquire) {
                            return Ok(());
                        }
                        first.get_or_insert(start.elapsed().as_secs_f64());
                        pcm.extend_from_slice(audio);
                        if let Some(output) = &mut playback {
                            output.push(audio).map_err(omni_breeze_tts::Error::Audio)?;
                        }
                        let stats = playback.as_ref().and_then(|p| p.stats());
                        let mut u = Update::status(
                            "audio",
                            if let Some(stats) = &stats {
                                if stats.buffering {
                                    format!(
                                        "正在缓冲 · 已准备 {:.1} 秒音频 · 可以随时停止",
                                        stats.buffered_seconds
                                    )
                                } else {
                                    "正在合成并播放 · 可以随时停止".into()
                                }
                            } else if hold {
                                "正在合成待选音频 · 可以随时停止".into()
                            } else if playback.is_some() {
                                "正在生成音频 · 可以随时停止".into()
                            } else {
                                "正在合成 · 无可用扬声器，完成后可保存 WAV".into()
                            },
                        );
                        u.audio_seconds = pcm.len() as f64 / SAMPLE_RATE as f64;
                        u.elapsed = start.elapsed().as_secs_f64();
                        u.first_audio = first;
                        u.playback = offset_playback(stats, playback_offset);
                        sender.publish(u);
                        Ok(())
                    })
                    .map_err(Failure::from)?;
                let elapsed = start.elapsed().as_secs_f64();
                if hold {
                    let end = if stop.load(Ordering::Acquire) || report.stopped {
                        enhancement::CandidateEnd::Stopped
                    } else if report.reached_limit {
                        enhancement::CandidateEnd::ReachedLimit
                    } else {
                        enhancement::CandidateEnd::Normal {
                            delivered_audio: !pcm.is_empty(),
                        }
                    };
                    let choosable = enhancement::candidate_is_choosable(end);
                    let mut held = Update::status(
                        if choosable { "held" } else { "done" },
                        if choosable {
                            "已合成一段待选音频"
                        } else {
                            "这一段没有正常结束 · 未完成的盲选已作废"
                        },
                    );
                    if choosable {
                        held.audio = Some(Arc::from(pcm.clone()));
                    }
                    held.audio_seconds = report.audio_seconds;
                    held.elapsed = elapsed;
                    held.first_audio = first;
                    return Ok(held);
                }
                let mut generated = Update::status("generated", "生成完成 · 可保存音频");
                if !pcm.is_empty() {
                    generated.audio = Some(Arc::from(pcm.clone()));
                }
                generated.audio_seconds = report.audio_seconds;
                context.release_inference();
                sender.publish(generated);
                if !report.stopped && !report.reached_limit {
                    duration_history.observe(voice, &options.text, report.audio_seconds);
                }
                if let Some(output) = &mut playback {
                    output.drain_with_progress(|stats| {
                        let mut u = Update::status(
                            "draining",
                            "生成完成 · 正在播放剩余音频 · 可以随时停止",
                        );
                        u.audio_seconds = pcm.len() as f64 / SAMPLE_RATE as f64;
                        u.elapsed = elapsed;
                        u.first_audio = first;
                        u.playback = offset_playback(Some(stats), playback_offset);
                        sender.publish(u);
                    })?;
                }
                let mut u = Update::status(
                    "done",
                    if stop.load(Ordering::Acquire) || report.stopped {
                        if pcm.is_empty() {
                            "已停止 · 尚未生成音频"
                        } else {
                            "已停止 · 可保存已生成音频"
                        }
                    } else if report.reached_limit {
                        "已达到本段 60 秒生成上限 · 请将剩余文本分段合成"
                    } else {
                        "合成完成 · 可保存 WAV"
                    },
                );
                u.audio_seconds = pcm.len() as f64 / SAMPLE_RATE as f64;
                // Include reference preparation in the user-visible latency.
                u.elapsed = elapsed;
                u.first_audio = first;
                u.playback =
                    offset_playback(playback.as_ref().and_then(|p| p.stats()), playback_offset);
                if design
                    && !stop.load(Ordering::Acquire)
                    && !report.stopped
                    && !report.reached_limit
                    && !pcm.is_empty()
                {
                    // Snapshot the text used for this waveform, rather than
                    // reading inputs that the user may have edited meanwhile.
                    *completed_design = Some(Design {
                        text: options.text.clone(),
                        seed: options.sampling.seed,
                    });
                    u.reusable = true;
                    u.design_seed = Some(options.sampling.seed);
                    u.message = "音色设计完成 · 可再次试听，或使用此音色朗读新文字".into();
                }
                // Close the old device stream before admitting the next job.
                drop(playback);
                Ok(u)
            }
            Command::Replay(pcm) => {
                if pcm.is_empty() {
                    return Err("还没有可试听的音频".into());
                }
                let mut output = Playback::open(stop.clone(), None)?;
                sender.publish(Update::status(
                    "replaying",
                    "正在试听已有音频 · 可以随时停止",
                ));
                for chunk in pcm.chunks(3840) {
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    output.push(chunk)?;
                }
                output.drain()?;
                let mut u = Update::status(
                    "replayed",
                    if stop.load(Ordering::Acquire) {
                        "试听已停止"
                    } else {
                        "试听结束"
                    },
                );
                u.playback = output.stats();
                drop(output);
                Ok(u)
            }
            Command::UseDesign => {
                let design = completed_design
                    .as_ref()
                    .ok_or("请先完整生成一个音色候选，再使用此音色")?;
                sender.publish(Update::status("reference", "正在准备已设计音色…"));
                if reference_encoder.is_none() && !stop.load(Ordering::Acquire) {
                    *reference_encoder = Some(
                        ReferenceEncoder::load_cancellable(
                            &model_path.join("audio_tokenizer"),
                            device,
                            &stop,
                        )
                        .map_err(Failure::from)?,
                    );
                }
                let encoded = if stop.load(Ordering::Acquire) {
                    None
                } else {
                    reference_encoder
                        .as_ref()
                        .unwrap()
                        .encode(&pcm, &stop, |p| {
                            let mut u = Update::status(
                                "reference",
                                format!("正在准备已设计音色 · {:.0}%", p * 100.0),
                            );
                            u.progress = p;
                            sender.publish(u);
                        })
                        .map_err(Failure::from)?
                };
                if let Some(encoded) = encoded {
                    *designed_reference = Some(ReferenceAudio {
                        text: design.text.clone(),
                        codes: encoded.codes,
                    });
                    let mut u =
                        Update::status("voice_ready", "已选用设计音色 · 修改朗读文字后即可合成");
                    u.reference_text = Some(design.text.clone());
                    u.design_seed = Some(design.seed);
                    Ok(u)
                } else {
                    Ok(Update::status(
                        "voice_cancelled",
                        "已取消准备音色 · 仍可试听或重新选用",
                    ))
                }
            }
        }
    }
}
