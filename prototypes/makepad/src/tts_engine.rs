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
    Model, Precision, ReferenceAudio, ScriptBudget, ScriptEvent, ScriptOptions, SynthesisOptions,
    SAMPLE_RATE,
};
use std::{
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
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

    fn stats(rendered: f64, first: Option<f64>, underruns: usize, gap: f64) -> PlaybackStats {
        PlaybackStats {
            buffered_seconds: 0.0,
            target_seconds: 8.0,
            rendered_seconds: rendered,
            first_playback_seconds: first,
            buffering: false,
            underruns,
            gap_seconds: gap,
            output_rate: 48_000,
        }
    }

    #[test]
    fn a_scripts_numbers_add_the_opening_to_the_rest_and_keep_the_first_sound() {
        let opening = stats(3.2, Some(4.5), 0, 0.0);
        let joined = join_playback(stats(10.0, Some(60.0), 1, 0.25), &opening);
        assert!((joined.rendered_seconds - 13.2).abs() < 1e-9);
        assert_eq!(joined.first_playback_seconds, Some(4.5));
        assert_eq!(joined.underruns, 1);
        assert!((joined.gap_seconds - 0.25).abs() < 1e-9);
        // Until the rest has sounded, the opening's first sound is the job's.
        let silent = stats(0.0, None, 0, 0.0);
        assert_eq!(
            join_playback(silent, &opening).first_playback_seconds,
            Some(4.5)
        );
    }

    #[test]
    fn the_rest_of_a_long_script_waits_for_its_lead_and_says_how_long() {
        let message = waiting_message(30.0, 270.0, 600.0, None, 0);
        assert!(
            message.contains("已准备 30 秒") && message.contains("攒够约 270 秒后连续播放"),
            "{message}"
        );
        assert!(!message.contains("预计"), "{message}");
        // 30 seconds of the rest took 20: the 240 still wanted take about 160.
        let started = Some((Instant::now() - std::time::Duration::from_secs(20), 0));
        let message = waiting_message(30.0, 270.0, 600.0, started, 30 * SAMPLE_RATE as usize);
        assert!(message.contains("预计还要等 2 分 40 秒"), "{message}");
        // More lead than the rest can ever hold: it plays once all is generated.
        let message = waiting_message(30.0, 700.0, 66.0, started, 0);
        assert!(
            message.contains("生成完成后连续播放") && !message.contains("攒够"),
            "{message}"
        );
        assert!(message.ends_with("可以随时停止"), "{message}");
    }

    #[test]
    fn the_status_names_waiting_then_buffering_then_playing() {
        let buffering = PlaybackStats {
            buffered_seconds: 1.5,
            target_seconds: 8.0,
            rendered_seconds: 0.0,
            first_playback_seconds: None,
            buffering: true,
            underruns: 0,
            gap_seconds: 0.0,
            output_rate: 48_000,
        };
        let playing = PlaybackStats {
            buffering: false,
            ..buffering
        };
        assert_eq!(
            progress_message(Some(&buffering), Some("等待".into()), false, true),
            "等待"
        );
        assert_eq!(
            progress_message(Some(&buffering), None, false, true),
            "正在缓冲 · 已准备 1.5 秒音频 · 可以随时停止"
        );
        assert_eq!(
            progress_message(Some(&playing), None, false, true),
            "正在合成并播放 · 可以随时停止"
        );
        assert_eq!(
            progress_message(None, None, true, false),
            "正在合成待选音频 · 可以随时停止"
        );
        assert_eq!(
            progress_message(None, None, false, true),
            "正在生成音频 · 可以随时停止"
        );
        assert_eq!(
            progress_message(None, None, false, false),
            "正在合成 · 无可用扬声器，完成后可保存 WAV"
        );
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
    /// How the opening speech text played, once a script moves on to the rest.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opening_playback: Option<PlaybackStats>,
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
            opening_playback: None,
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
fn running(
    message: String,
    audio_seconds: f64,
    elapsed: f64,
    first: Option<f64>,
    playback: Option<PlaybackStats>,
) -> Update {
    let mut update = Update::status("audio", message);
    update.audio_seconds = audio_seconds;
    update.elapsed = elapsed;
    update.first_audio = first;
    update.playback = playback;
    update
}
fn labelled(segment: &Option<String>, message: String) -> String {
    match segment {
        Some(segment) => format!("{segment}{message}"),
        None => message,
    }
}

/// Device output for one script. The opening speech text plays as it arrives.
/// Every later speech text waits until the playback policy has banked enough to
/// run without stopping, then plays on a stream of its own.
struct ScriptOutput {
    opening: Option<Playback>,
    /// Opened once the opening has been played to its end.
    rest: Option<Playback>,
    /// Job-relative clock zeros of the two streams.
    offsets: (f64, f64),
    /// Final numbers of the opening stream, on the job's clock.
    opening_stats: Option<PlaybackStats>,
}
impl ScriptOutput {
    fn open(stop: Arc<AtomicBool>, estimate: f64, offset: f64) -> Result<Self, String> {
        Ok(Self {
            opening: Some(Playback::open(stop, Some(estimate))?),
            rest: None,
            offsets: (offset, offset),
            opening_stats: None,
        })
    }
    /// The opening speech text is complete: let it finish and close its stream,
    /// then open the one for the rest, whose ring is sized for its estimate.
    /// `started` is the job's clock zero. `Ok(Some(message))` means the rest
    /// could not be opened.
    fn start_rest(
        &mut self,
        stop: Arc<AtomicBool>,
        estimate: f64,
        started: Instant,
        mut progress: impl FnMut(PlaybackStats),
    ) -> Result<Option<String>, String> {
        let offset_of_opening = self.offsets.0;
        if let Some(mut opening) = self.opening.take() {
            opening.drain_with_progress(|stats| {
                progress(offset_playback(Some(stats), offset_of_opening).unwrap());
            })?;
            self.opening_stats = offset_playback(opening.stats(), offset_of_opening);
        }
        self.rest = match Playback::open(stop, Some(estimate)) {
            Ok(rest) => Some(rest),
            Err(message) => return Ok(Some(message)),
        };
        // Taken once the stream exists: listening to the opening came first.
        self.offsets.1 = started.elapsed().as_secs_f64();
        Ok(None)
    }
    fn push(&mut self, audio: &[f32]) -> Result<(), String> {
        match self.rest.as_mut() {
            Some(rest) => rest.push(audio),
            None => match self.opening.as_mut() {
                Some(opening) => opening.push(audio),
                None => Ok(()),
            },
        }
    }
    /// The rest is being generated but does not play yet: (banked, wanted).
    fn waiting(&self) -> Option<(f64, f64)> {
        let stats = self.rest.as_ref()?.stats()?;
        stats
            .buffering
            .then_some((stats.buffered_seconds, stats.target_seconds))
    }
    /// The whole job so far on the job's clock; None when output is silent.
    fn stats(&self) -> Option<PlaybackStats> {
        if let Some(rest) =
            offset_playback(self.rest.as_ref().and_then(Playback::stats), self.offsets.1)
        {
            return Some(match &self.opening_stats {
                Some(opening) => join_playback(rest, opening),
                None => rest,
            });
        }
        offset_playback(
            self.opening.as_ref().and_then(Playback::stats),
            self.offsets.0,
        )
    }
    /// How the opening speech text played, once the script has moved on.
    fn opening_playback(&self) -> Option<PlaybackStats> {
        self.opening_stats.clone()
    }
    fn drain(&mut self, mut progress: impl FnMut(PlaybackStats)) -> Result<(), String> {
        if let Some(rest) = &mut self.rest {
            let offset = self.offsets.1;
            let opening = self.opening_stats.clone();
            rest.drain_with_progress(|stats| {
                let stats = offset_playback(Some(stats), offset).unwrap();
                progress(match &opening {
                    Some(opening) => join_playback(stats, opening),
                    None => stats,
                });
            })
        } else if let Some(opening) = &mut self.opening {
            let offset = self.offsets.0;
            opening.drain_with_progress(|stats| {
                progress(offset_playback(Some(stats), offset).unwrap());
            })
        } else {
            Ok(())
        }
    }
}

/// The rest's live numbers on top of the opening's final ones.
fn join_playback(mut rest: PlaybackStats, opening: &PlaybackStats) -> PlaybackStats {
    rest.rendered_seconds += opening.rendered_seconds;
    rest.underruns += opening.underruns;
    rest.gap_seconds += opening.gap_seconds;
    if rest.first_playback_seconds.is_none_or(|rest| {
        opening
            .first_playback_seconds
            .is_some_and(|first| first < rest)
    }) {
        rest.first_playback_seconds = opening.first_playback_seconds;
    }
    rest
}

/// Status while the rest of a script is banked before it plays.
fn waiting_message(
    banked: f64,
    wanted: f64,
    estimate: f64,
    started: Option<(Instant, usize)>,
    generated: usize,
) -> String {
    let mut message = format!("正在准备后续音频 · 已准备 {banked:.0} 秒");
    if wanted < estimate {
        message.push_str(&format!(" · 攒够约 {wanted:.0} 秒后连续播放"));
        // The first chunk still carries the prefill, so this waits too long.
        if let Some((since, before)) = started {
            let seconds = (generated - before) as f64 / SAMPLE_RATE as f64;
            let elapsed = since.elapsed().as_secs_f64();
            if seconds > 0.0 && elapsed > 0.0 {
                let wait = ((wanted - banked).max(0.0) / (seconds / elapsed)) as u64;
                message.push_str(&format!(" · 预计还要等 {} 分 {} 秒", wait / 60, wait % 60));
            }
        }
    } else {
        message.push_str(" · 生成完成后连续播放");
    }
    message.push_str(" · 可以随时停止");
    message
}

/// The status line while the rest of a script is banked before it plays.
fn waiting_line(
    output: Option<&ScriptOutput>,
    estimate: f64,
    started: Option<(Instant, usize)>,
    generated: usize,
) -> Option<String> {
    output?
        .waiting()
        .map(|(banked, wanted)| waiting_message(banked, wanted, estimate, started, generated))
}

fn progress_message(
    stats: Option<&PlaybackStats>,
    waiting: Option<String>,
    hold: bool,
    output: bool,
) -> String {
    match (waiting, stats) {
        (Some(waiting), _) => waiting,
        (None, Some(stats)) if stats.buffering => {
            format!(
                "正在缓冲 · 已准备 {:.1} 秒音频 · 可以随时停止",
                stats.buffered_seconds
            )
        }
        (None, Some(_)) => "正在合成并播放 · 可以随时停止".into(),
        (None, None) if hold => "正在合成待选音频 · 可以随时停止".into(),
        (None, None) if output => "正在生成音频 · 可以随时停止".into(),
        (None, None) => "正在合成 · 无可用扬声器，完成后可保存 WAV".into(),
    }
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
                // A long script is spoken as several speech texts, prepared
                // once: the estimate below and the synthesis share this split.
                // Without a reference recording they would each pick their own
                // voice, so the encoder that records the opening as their
                // shared reference is loaded before the first audio.
                let script = ScriptOptions {
                    synthesis: options.clone(),
                    budget: ScriptBudget::default(),
                };
                let prepared = model.prepare_script(&script).map_err(Failure::from)?;
                let speech_texts = prepared.texts.len();
                if prepared.needs_voice_anchor && reference_encoder.is_none() {
                    sender.publish(Update::status("reference", "首次加载参考音频编码器…"));
                    *reference_encoder = Some(
                        ReferenceEncoder::load_cancellable(
                            &model_path.join("audio_tokenizer"),
                            device,
                            stop,
                        )
                        .map_err(Failure::from)?,
                    );
                }
                // Keep synthesis/save usable on machines with no audio output.
                let playback_offset = start.elapsed().as_secs_f64();
                let voice = duration_voice(&options);
                let ceiling = options.max_frames as f64 * 0.08;
                let estimate = |text: &str, speech_texts: usize| {
                    duration_history
                        .estimate(voice, text)
                        .min(ceiling * speech_texts as f64)
                };
                let opening_estimate = estimate(&prepared.texts[0].text, 1);
                let rest_estimate =
                    (estimate(&options.text, speech_texts) - opening_estimate).max(1.0);
                // The opening speech text plays as it arrives. The rest waits on
                // its own stream until enough is banked to play without stopping.
                let mut playback = if hold {
                    None
                } else {
                    match ScriptOutput::open(stop.clone(), opening_estimate, playback_offset) {
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
                // When the rest began to be generated: its clock zero and the
                // audio banked by then, for the wait estimate in the status.
                let mut rest_started = None;
                // Which speech text is running; a single one needs no label.
                let mut segment = (speech_texts > 1).then(|| format!("第 1/{speech_texts} 段 · "));
                let mut first = None;
                let report = model
                    .speak_prepared(prepared, &stop, reference_encoder.as_ref(), |event| {
                        if stop.load(Ordering::Acquire) {
                            return Ok(());
                        }
                        first.get_or_insert(start.elapsed().as_secs_f64());
                        let seconds = pcm.len() as f64 / SAMPLE_RATE as f64;
                        let waiting =
                            waiting_line(playback.as_ref(), rest_estimate, rest_started, pcm.len());
                        match event {
                            ScriptEvent::Audio(audio) => {
                                pcm.extend_from_slice(audio);
                                if let Some(output) = &mut playback {
                                    output.push(audio).map_err(omni_breeze_tts::Error::Audio)?;
                                }
                                let stats = playback.as_ref().and_then(ScriptOutput::stats);
                                let message = progress_message(
                                    stats.as_ref(),
                                    waiting,
                                    hold,
                                    playback.is_some(),
                                );
                                let update = running(
                                    labelled(&segment, message),
                                    pcm.len() as f64 / SAMPLE_RATE as f64,
                                    start.elapsed().as_secs_f64(),
                                    first,
                                    stats,
                                );
                                sender.publish(update);
                            }
                            ScriptEvent::SpeechText(index, _) => {
                                // The opening has been generated: it finishes playing
                                // on its stream, and the rest continues on a new one.
                                let rest_follows = index == 0 && speech_texts > 1;
                                if rest_follows {
                                    if let Some(output) = &mut playback {
                                        let label = labelled(
                                            &segment,
                                            "开头一段已生成 · 正在听完 · 可以随时停止".to_owned(),
                                        );
                                        sender.publish(running(
                                            label.clone(),
                                            seconds,
                                            start.elapsed().as_secs_f64(),
                                            first,
                                            output.stats(),
                                        ));
                                        let (sender, started, first) = (sender, start, first);
                                        let notice = output
                                            .start_rest(
                                                stop.clone(),
                                                rest_estimate,
                                                started,
                                                move |stats| {
                                                    sender.publish(running(
                                                        label.clone(),
                                                        seconds,
                                                        started.elapsed().as_secs_f64(),
                                                        first,
                                                        Some(stats),
                                                    ));
                                                },
                                            )
                                            .map_err(omni_breeze_tts::Error::Audio)?;
                                        if let Some(notice) = notice {
                                            sender.publish(Update::status(
                                                "notice",
                                                format!("{notice} · 仍可合成并保存 WAV"),
                                            ));
                                        }
                                        rest_started = Some((Instant::now(), pcm.len()));
                                    }
                                }
                                segment = Some(format!("第 {}/{speech_texts} 段 · ", index + 2));
                                if rest_follows {
                                    // First look at the rest. No audio of it has
                                    // arrived yet, so the policy has no lead to name.
                                    let stats = playback.as_ref().and_then(ScriptOutput::stats);
                                    let message = progress_message(
                                        stats.as_ref(),
                                        None,
                                        hold,
                                        playback.is_some(),
                                    );
                                    sender.publish(running(
                                        labelled(&segment, message),
                                        seconds,
                                        start.elapsed().as_secs_f64(),
                                        first,
                                        stats,
                                    ));
                                }
                            }
                            ScriptEvent::Anchor(count) => {
                                let stats = playback.as_ref().and_then(ScriptOutput::stats);
                                let message =
                                    format!("已从开头 {count} 段录下参考 · 后续各段沿用同一音色");
                                sender.publish(running(
                                    labelled(&segment, message),
                                    seconds,
                                    start.elapsed().as_secs_f64(),
                                    first,
                                    stats,
                                ));
                            }
                        }
                        Ok(())
                    })
                    .map_err(Failure::from)?;
                let elapsed = start.elapsed().as_secs_f64();
                if hold {
                    let end = if stop.load(Ordering::Acquire) || report.stopped {
                        enhancement::CandidateEnd::Stopped
                    } else if report.reached_limit.is_some() {
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
                if !report.stopped && report.reached_limit.is_none() {
                    duration_history.observe(voice, &options.text, report.audio_seconds);
                }
                if let Some(output) = &mut playback {
                    output.drain(|stats| {
                        let mut u = Update::status(
                            "draining",
                            "生成完成 · 正在播放剩余音频 · 可以随时停止",
                        );
                        u.audio_seconds = pcm.len() as f64 / SAMPLE_RATE as f64;
                        u.elapsed = elapsed;
                        u.first_audio = first;
                        u.playback = Some(stats);
                        sender.publish(u);
                    })?;
                }
                let mut u = Update::status(
                    "done",
                    if stop.load(Ordering::Acquire) || report.stopped {
                        if pcm.is_empty() {
                            "已停止 · 尚未生成音频".into()
                        } else {
                            "已停止 · 可保存已生成音频".into()
                        }
                    } else if let Some(index) = report.reached_limit {
                        // A speech text that reaches the ceiling leaves its unread
                        // words unlocatable, so the script stops there.
                        let seconds = options.max_frames * 8 / 100;
                        if speech_texts > 1 {
                            format!(
                                "第 {} 段达到单段 {seconds} 秒生成上限 · 其后 {} 段未合成 · 可保存已生成音频",
                                index + 1,
                                speech_texts - index - 1
                            )
                        } else {
                            format!("已达到单段 {seconds} 秒生成上限 · 请缩短文本后重试")
                        }
                    } else {
                        "合成完成 · 可保存 WAV".into()
                    },
                );
                u.audio_seconds = pcm.len() as f64 / SAMPLE_RATE as f64;
                // Include reference preparation in the user-visible latency.
                u.elapsed = elapsed;
                u.first_audio = first;
                u.playback = playback.as_ref().and_then(ScriptOutput::stats);
                u.opening_playback = playback.as_ref().and_then(ScriptOutput::opening_playback);
                if design
                    && !stop.load(Ordering::Acquire)
                    && !report.stopped
                    && report.reached_limit.is_none()
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
