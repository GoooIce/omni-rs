use crate::jobs::{Context, Executor, Output};
use omni_r2t2::{Precision, StreamConfig};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};

#[cfg(feature = "metal")]
type Backend = burn::backend::wgpu::Metal;
#[cfg(all(not(feature = "metal"), feature = "webgpu"))]
type Backend = burn::backend::wgpu::WebGpu;
#[cfg(all(not(feature = "metal"), not(feature = "webgpu"), feature = "cpu"))]
type Backend = burn::backend::NdArray;

#[derive(Clone)]
pub struct ModelPaths {
    pub asr: PathBuf,
    pub tts: PathBuf,
    pub translation: PathBuf,
    pub image: PathBuf,
    pub music: PathBuf,
    pub output: PathBuf,
    pub precision: Precision,
    pub cache_tokens: usize,
    pub translation_batch: omni_hy_mt::BatchConfig,
    pub asr_batch_size: usize,
    pub tts_batch_size: usize,
    pub image_batch_size: usize,
    pub asr_paged_attention: bool,
    pub tts_paged_attention: bool,
}
pub enum Command {
    Translate(omni_hy_mt::Request),
    Transcribe {
        audio: Vec<u8>,
        config: StreamConfig,
    },
    Speech(Speech),
    Image(Image),
    Music(omni_music::pipeline::Request),
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Speech {
    pub text: String,
    pub instruction: String,
    pub seed: u64,
    pub max_frames: usize,
    pub guidance_scale: f32,
}
impl Default for Speech {
    fn default() -> Self {
        Self {
            text: String::new(),
            instruction: String::new(),
            seed: 42,
            max_frames: 750,
            guidance_scale: 1.0,
        }
    }
}
impl Speech {
    pub fn validate(&self) -> Result<(), String> {
        if self.text.trim().is_empty()
            || self.text.len() > 32768
            || self.instruction.len() > 8192
            || !(1..=1500).contains(&self.max_frames)
            || !self.guidance_scale.is_finite()
            || !(1.0..=10.0).contains(&self.guidance_scale)
        {
            return Err("invalid speech text or generation limits".into());
        }
        Ok(())
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Image {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    pub true_cfg_scale: f32,
}
impl Default for Image {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: None,
            width: 512,
            height: 512,
            seed: 42,
            true_cfg_scale: 1.0,
        }
    }
}
impl Image {
    pub fn options(&self) -> omni_qwen_image::Options {
        omni_qwen_image::Options {
            prompt: self.prompt.clone(),
            negative_prompt: self.negative_prompt.clone(),
            width: self.width,
            height: self.height,
            seed: self.seed,
            true_cfg_scale: self.true_cfg_scale,
            ..Default::default()
        }
    }
}
#[derive(Debug)]
pub enum Failure {
    Cancelled,
    Error(String),
}
impl From<omni_r2t2::Error> for Failure {
    fn from(e: omni_r2t2::Error) -> Self {
        match e {
            omni_r2t2::Error::Interrupted => Self::Cancelled,
            _ => Self::Error(e.to_string()),
        }
    }
}
impl From<omni_qwen_image::Error> for Failure {
    fn from(e: omni_qwen_image::Error) -> Self {
        match e {
            omni_qwen_image::Error::Cancelled => Self::Cancelled,
            _ => Self::Error(e.to_string()),
        }
    }
}
impl From<std::io::Error> for Failure {
    fn from(e: std::io::Error) -> Self {
        Self::Error(e.to_string())
    }
}

mod driver;
mod music;
use driver::{Driver, NativeDriver, Product, Step, Update};

impl driver::Progress {
    pub fn value(self) -> serde_json::Value {
        use serde_json::json;
        match self {
            Self::Loading { progress, message } => {
                json!({"stage":"loading","progress":progress,"message":message})
            }
            Self::Tensors { current, total } => {
                json!({"stage":"loading","current":current,"total":total})
            }
            Self::Translation(e) => {
                json!({"stage":"translation","text":e.text,"generated_tokens":e.generated_tokens})
            }
            Self::Transcription(e) => {
                json!({"stage":"transcription","text":e.event.text,"audio_seconds":e.event.audio_seconds})
            }
            Self::Image(e) => json!(e),
            Self::Music(e) => json!(e),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Kind {
    Translation,
    Transcription,
    Speech,
    Image,
    Music,
}
impl Command {
    fn kind(&self) -> Kind {
        match self {
            Self::Translate(_) => Kind::Translation,
            Self::Transcribe { .. } => Kind::Transcription,
            Self::Speech(_) => Kind::Speech,
            Self::Image(_) => Kind::Image,
            Self::Music(_) => Kind::Music,
        }
    }
}
struct JobRun {
    context: Context,
    audio: Vec<f32>,
}
struct Active {
    kind: Kind,
    members: BTreeMap<u64, JobRun>,
    failure: Option<String>,
}
/// Resident weights survive modality switches; only one modality owns active jobs.
/// The driver receives cancellation and typed messages, never a job Context.
pub(crate) struct Models<D: Driver = NativeDriver> {
    driver: D,
    resident: BTreeMap<Kind, D::Loaded>,
    active: Option<Active>,
}
impl Models {
    pub fn new(paths: ModelPaths) -> Self {
        Self::with_driver(NativeDriver::new(paths))
    }
}
impl<D: Driver> Models<D> {
    fn with_driver(driver: D) -> Self {
        Self {
            driver,
            resident: BTreeMap::new(),
            active: None,
        }
    }
    pub fn unload(&mut self) {
        self.resident.clear();
        self.active = None;
    }
    fn fail_active(&mut self, message: String) -> Vec<(String, Result<Output, Failure>)> {
        let active = self.active.take();
        // Drop every engine before returning terminal results, including retained
        // weights whose state may have been invalidated by a protocol failure.
        self.resident.clear();
        active
            .into_iter()
            .flat_map(|active| active.members.into_values())
            .map(|job| (job.context.id(), Err(Failure::Error(message.clone()))))
            .collect()
    }
    fn apply(&mut self, step: Step) -> Vec<(String, Result<Output, Failure>)> {
        let Some(active) = self.active.as_ref() else {
            return Vec::new();
        };
        // Validate the entire step before emitting progress or preparing files.
        // A duplicate/unknown completion must never strand another running job.
        let mut seen = std::collections::BTreeSet::new();
        let invalid = step.updates.iter().any(|(id, update)| {
            !active.members.contains_key(id) || update.kind() != Some(active.kind)
        }) || step.completed.iter().any(|(id, product)| {
            !active.members.contains_key(id)
                || !seen.insert(*id)
                || product.as_ref().is_ok_and(|p| p.kind() != active.kind)
        });
        if invalid {
            return self.fail_active(
                "model driver protocol error: unknown, duplicate, or mismatched request event"
                    .into(),
            );
        }
        let active = self.active.as_mut().unwrap();
        for (id, update) in step.updates {
            let job = active.members.get_mut(&id).unwrap();
            match update {
                Update::Progress(progress) => job.context.progress(progress.value()),
                Update::Audio(audio) => {
                    job.audio.extend(audio);
                    job.context.progress(json!({"stage":"speech","audio_seconds":job.audio.len() as f64 / omni_breeze_tts::SAMPLE_RATE as f64}));
                }
            }
        }
        let completed = step
            .completed
            .into_iter()
            .map(|(id, result)| {
                let job = active.members.remove(&id).unwrap();
                let result = result.and_then(|product| match product {
                    Product::Translation { report, batch } => {
                        let cancelled = report.finish_reason == "cancelled";
                        let mut value = json!(report);
                        value["batch"] = json!(batch);
                        Ok(Output::new(value, cancelled))
                    }
                    Product::Transcription { report, batch } => {
                        let mut value = json!(report);
                        value["batch"] = json!(batch);
                        Ok(Output::new(value, report.cancelled))
                    }
                    Product::Speech { report, batch } => {
                        speech_output(&job.context, job.audio, report, batch)
                    }
                    Product::Image { encode, batch } => image_output(&job.context, encode, batch),
                    Product::Music(product) => music_output(&job.context, product),
                });
                (job.context.id(), result)
            })
            .collect();
        if active.members.is_empty() {
            self.active = None;
        }
        completed
    }
}
impl<D: Driver> Executor for Models<D> {
    type Command = Command;
    fn recover(&mut self) {
        self.unload();
    }
    fn batchable(&self, command: &Command) -> bool {
        self.active
            .as_ref()
            .is_none_or(|active| active.kind == command.kind())
    }
    fn try_start(&mut self, command: &Command, context: &Context) -> Result<bool, Failure> {
        // Enforce the invariant here too: callers need not preflight batchable.
        if !self.batchable(command) {
            return Ok(false);
        }
        if let Some(message) = self.active.as_ref().and_then(|a| a.failure.as_ref()) {
            return Err(Failure::Error(message.clone()));
        }
        let kind = command.kind();
        if let std::collections::btree_map::Entry::Vacant(entry) = self.resident.entry(kind) {
            let loaded = self
                .driver
                .load(kind, context.cancel(), &|p| context.progress(p.value()))?;
            entry.insert(loaded);
        }
        let engine = self.resident.get_mut(&kind).unwrap();
        let Some(id) = self
            .driver
            .admit(engine, command, context.cancel().clone(), &|p| {
                context.progress(p.value())
            })?
        else {
            return Ok(false);
        };
        let active = self.active.get_or_insert_with(|| Active {
            kind,
            members: BTreeMap::new(),
            failure: None,
        });
        if active.members.contains_key(&id) {
            let message = "model driver protocol error: duplicate admitted request ID".to_string();
            active.failure = Some(message.clone());
            // The failed admission may already own engine resources. Release
            // them before Jobs can publish this new request's error; the next
            // step returns the retained older jobs' failures.
            self.resident.clear();
            return Err(Failure::Error(message));
        }
        active.members.insert(
            id,
            JobRun {
                context: context.clone(),
                audio: Vec::new(),
            },
        );
        Ok(true)
    }
    fn step(&mut self) -> Vec<(String, Result<Output, Failure>)> {
        let Some(active) = &mut self.active else {
            return Vec::new();
        };
        if let Some(message) = active.failure.take() {
            return self.fail_active(message);
        }
        let Some(engine) = self.resident.get_mut(&active.kind) else {
            return self
                .fail_active("model driver protocol error: active model is not resident".into());
        };
        // End the engine borrow before applying owned progress, PCM and outputs.
        let step = self.driver.step(engine);
        self.apply(step)
    }
    fn run(&mut self, _command: Command, _context: &Context) -> Result<Output, Failure> {
        Err(Failure::Error("request requires batch admission".into()))
    }
}

fn image_output(
    job: &Context,
    encode: driver::Png,
    batch: omni_qwen_image::BatchStats,
) -> Result<Output, Failure> {
    let filename = format!("{}.png", job.id());
    let mut report = None;
    let file = job.file("png", |path| {
        report = Some(encode(path)?);
        Ok(())
    })?;
    let mut output = Output::new(
        json!({"image_url":format!("/v1/files/{filename}"),"report":report,"batch":batch}),
        false,
    );
    output.file = Some(file);
    Ok(output)
}

fn music_output(job: &Context, product: omni_music::pipeline::Product) -> Result<Output, Failure> {
    use omni_music::pipeline::Product;
    let url = |ext: &str| format!("/v1/files/{}.{ext}", job.id());
    let mut value = match &product {
        Product::Analysis(report) => json!(report),
        Product::Score(score) => {
            let mut value = json!(score);
            value["midi_url"] = json!(url("mid"));
            if score.abc.is_some() {
                value["abc_url"] = json!(url("abc"));
            }
            value
        }
        Product::Song {
            audio,
            semantic,
            elapsed_seconds,
        } => {
            let mut value = json!({"model":"m-a-p/YuE2-3B","sample_rate":48000,"channels":2,
                "audio_seconds":audio.len() as f64/96000.0,"audio_url":url("wav"),
                "abc":semantic.plan.abc,"semantic_tokens":semantic.tokens,
                "request":semantic.plan.request,"compute_seconds":elapsed_seconds,
                "truncated":semantic.truncated||semantic.plan.truncated,
                "score_truncated":semantic.plan.truncated,"music_truncated":semantic.truncated,
                "backend":"burn"});
            if semantic.plan.abc.is_some() {
                value["abc_url"] = json!(url("abc"));
            }
            value
        }
    };
    value["result_url"] = json!(url("json"));
    let file = job.file("json", |path| {
        match &product {
            Product::Analysis(_) => {}
            Product::Score(score) => {
                std::fs::write(path.with_extension("mid"), score.midi())?;
                if let Some(abc) = &score.abc {
                    std::fs::write(path.with_extension("abc"), abc)?;
                }
            }
            Product::Song {
                audio, semantic, ..
            } => {
                omni_music::audio::write_wav(&path.with_extension("wav"), audio, 48000, 2)?;
                if let Some(abc) = &semantic.plan.abc {
                    std::fs::write(path.with_extension("abc"), abc)?;
                }
            }
        }
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&value).map_err(|e| Failure::Error(e.to_string()))?,
        )?;
        Ok(())
    })?;
    let mut output = Output::new(value, false);
    output.file = Some(file);
    Ok(output)
}

fn speech_output(
    job: &Context,
    samples: Vec<f32>,
    report: omni_breeze_tts::SynthesisReport,
    batch: omni_breeze_tts::BatchStats,
) -> Result<Output, Failure> {
    let filename = format!("{}.wav", job.id());
    let file = job.file("wav", |path| {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: omni_breeze_tts::SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer =
            hound::WavWriter::create(path, spec).map_err(|e| Failure::Error(e.to_string()))?;
        for value in samples {
            writer
                .write_sample((value.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16)
                .map_err(|e| Failure::Error(e.to_string()))?;
        }
        writer
            .finalize()
            .map_err(|e| Failure::Error(e.to_string()))?;
        Ok(())
    })?;
    let mut output = Output::new(
        json!({"audio_url":format!("/v1/files/{filename}"),"audio_seconds":report.audio_seconds,
        "cancelled":report.stopped,"reached_limit":report.reached_limit,"compute_seconds":report.elapsed_seconds,"first_audio_seconds":report.first_audio_seconds,"batch":batch}),
        report.stopped,
    );
    output.file = Some(file);
    Ok(output)
}

#[cfg(test)]
mod tests;
