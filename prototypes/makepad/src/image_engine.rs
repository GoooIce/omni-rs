use omni_qwen_image::{
    Config, Error, GeneratedImage, Model, ObservationRow, Options, Precision, Progress,
};
use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

// The measured speech policy is independent of the new diffusion pipeline.
// Use the existing process-wide Metal registration, with all image tensors on GPU.
#[cfg(target_os = "macos")]
type Backend = burn::backend::wgpu::Metal;
#[cfg(not(target_os = "macos"))]
type Backend = burn::backend::wgpu::WebGpu;

pub enum Command {
    Load(PathBuf),
    References(Vec<PathBuf>),
    Generate(Options),
}

pub struct ReferencePreview {
    pub paths: Vec<PathBuf>,
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u32>,
}

pub struct ObservationPreview {
    pub row: ObservationRow,
    pub step: usize,
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

#[derive(serde::Serialize)]
pub struct Update {
    pub kind: &'static str,
    pub message: String,
    pub progress: f64,
    #[serde(skip)]
    pub image: Option<Arc<GeneratedImage>>,
    #[serde(skip)]
    pub reference: Option<Arc<ReferencePreview>>,
    #[serde(skip)]
    pub observation: Option<ObservationPreview>,
}
impl Update {
    fn status(kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            progress: 0.0,
            image: None,
            reference: None,
            observation: None,
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

pub type Engine = crate::engine_harness::Engine<Worker>;

pub struct Worker {
    device: burn::backend::wgpu::WgpuDevice,
    model: Option<Model<Backend>>,
}

impl crate::engine_harness::Adapter for Worker {
    type Command = Command;
    type Update = Update;
    type Error = Error;
    const NAME: &'static str = "qwen-image";

    fn uses_model(command: &Command) -> bool {
        !matches!(command, Command::References(_))
    }
    fn new() -> Self {
        Self {
            device: Default::default(),
            model: None,
        }
    }
    fn unload(&mut self) {
        self.model = None;
    }
    fn load_path(command: &Command) -> Option<&Path> {
        if let Command::Load(path) = command {
            Some(path)
        } else {
            None
        }
    }
    fn check_files(path: &Path) -> Result<(), String> {
        Config::load(path).map_err(|e| e.to_string())?;
        crate::model_location::require_files(path, &["processor/tokenizer.json"])?;
        for (directory, name) in [
            ("text_encoder", "model"),
            ("transformer", "diffusion_pytorch_model"),
            ("vae", "diffusion_pytorch_model"),
        ] {
            crate::model_location::check_weights(&path.join(directory), name)?;
        }
        Ok(())
    }
    fn is_cancelled(error: &Error) -> bool {
        matches!(error, Error::Cancelled)
    }
    fn load(
        &mut self,
        path: &Path,
        context: &crate::engine_harness::Context<Update>,
    ) -> omni_qwen_image::Result<Update> {
        let stop = &context.stop;
        let start = Instant::now();
        self.model = Some(Model::load(
            &path,
            Precision::F16,
            &self.device,
            &stop,
            |p| {
                let (offset, weight, label) = match p.stage {
                    "load_text" => (0.0, 0.50, "文本编码器"),
                    "load_dit" => (0.50, 0.46, "图像生成主干"),
                    "load_vae_encoder" => (0.96, 0.02, "图像编码器"),
                    _ => (0.98, 0.02, "图像解码器"),
                };
                context.loading(
                    offset + weight * p.current as f64 / p.total.max(1) as f64,
                    format!("加载{label} · {} / {}", p.current, p.total),
                );
            },
        )?);
        let mut ready = Update::status(
            "ready",
            format!(
                "Qwen-Image-2.1-Turbo 已就绪 · FP16 · 加载 {:.1} s",
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
    ) -> omni_qwen_image::Result<Update> {
        let stop = &context.stop;
        match command {
            Command::Load(_) => unreachable!("load is handled by the harness"),
            Command::References(paths) => {
                use omni_qwen_image::reference;
                if paths.is_empty() || paths.len() > reference::MAX_REFERENCES {
                    return Err(Error::Config("请选择 1–4 张参考图片".into()));
                }
                let mut preview = None;
                for path in &paths {
                    let image = reference::open(path)?;
                    reference::dimensions(image.width(), image.height(), 1024)?;
                    if preview.is_none() {
                        let scale = 160.0 / image.width().max(image.height()) as f64;
                        let w = (image.width() as f64 * scale).round().max(1.0) as usize;
                        let h = (image.height() as f64 * scale).round().max(1.0) as usize;
                        let thumb = reference::resize(&image, w, h, &stop)?;
                        let pixels = thumb
                            .enumerate_pixels()
                            .map(|(x, y, p)| {
                                let matte = if (x / 8 + y / 8) % 2 == 0 {
                                    225u32
                                } else {
                                    195
                                };
                                let a = p[3] as u32;
                                let c = |v: u8| (v as u32 * a + matte * (255 - a) + 127) / 255;
                                0xff000000 | (c(p[0]) << 16) | (c(p[1]) << 8) | c(p[2])
                            })
                            .collect();
                        preview = Some((w, h, pixels));
                    }
                }
                let (width, height, pixels) = preview.unwrap();
                let mut update = Update::status(
                    "reference_ready",
                    format!("已选择 {} 张参考图 · 请填写修改要求", paths.len()),
                );
                update.reference = Some(Arc::new(ReferencePreview {
                    paths,
                    width,
                    height,
                    pixels,
                }));
                Ok(update)
            }
            Command::Generate(options) => {
                if options.observation {
                    return self.observe(options, context);
                }
                let model = self
                    .model
                    .as_mut()
                    .ok_or_else(|| Error::Config("图像模型尚未载入".into()))?;
                let total = model.config.sigmas.len();
                let editing = !options.images.is_empty();
                let mut track = ProgressTrack::default();
                let generated = Arc::new(model.generate(&options, &stop, |p: Progress| {
                    report_progress(context, &options, total, &mut track, p);
                })?);
                let mut update = Update::status(
                    "done",
                    format!(
                        "{}完成 · {} × {} · {:.1} s",
                        if editing { "编辑" } else { "生成" },
                        generated.report.width,
                        generated.report.height,
                        generated.report.elapsed_seconds
                    ),
                );
                update.progress = 1.0;
                update.image = Some(generated.clone());
                Ok(update)
            }
        }
    }
}

impl Worker {
    fn observe(
        &mut self,
        options: Options,
        context: &crate::engine_harness::Context<Update>,
    ) -> omni_qwen_image::Result<Update> {
        let stop = &context.stop;
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| Error::Config("图像模型尚未载入".into()))?;
        let total = model.config.sigmas.len();
        let track = RefCell::new(ProgressTrack::default());
        let observation = model.observe(
            &options,
            stop,
            |p: Progress| {
                report_progress(context, &options, total, &mut track.borrow_mut(), p);
            },
            |row, step, image| {
                let name = match row {
                    ObservationRow::Official => "官方文本条件",
                    ObservationRow::MiniCpm1b => "MiniCPM-V-4.7-1B 投影",
                    ObservationRow::MiniCpm35b => "MiniCPM-V-4.7-35B-A3B-int4 投影",
                };
                let mut update = Update::status(
                    "observation",
                    format!("去噪观察 · {name} · 第 {step} / {total} 步"),
                );
                update.progress = track.borrow().fraction_seen;
                update.observation = Some(ObservationPreview {
                    row,
                    step,
                    width: image.width() as usize,
                    height: image.height() as usize,
                    rgba: image.as_raw().to_vec(),
                });
                context.updates.publish(update);
            },
        )?;
        let (width, height) = observation.trajectories()[0].canvas();
        let official = observation.cells(ObservationRow::Official).len();
        let minicpm = observation.cells(ObservationRow::MiniCpm1b).len();
        let large = observation.cells(ObservationRow::MiniCpm35b).len();
        let mut update = Update::status(
            "done",
            format!(
                "去噪观察完成 · 官方 {official} 格 · MiniCPM-V-4.7-1B {minicpm} 格 · MiniCPM-V-4.7-35B-A3B-int4 {large} 格 · {width} × {height}"
            ),
        );
        update.progress = 1.0;
        Ok(update)
    }
}

#[derive(Default)]
struct ProgressTrack {
    step: usize,
    fraction_seen: f64,
    reference_index: usize,
}

fn report_progress(
    context: &crate::engine_harness::Context<Update>,
    options: &Options,
    total: usize,
    track: &mut ProgressTrack,
    p: Progress,
) {
    let branches = if options.guidance_enabled() { 2.0 } else { 1.0 };
    let editing = !options.images.is_empty();
    let base = if editing { 0.25 } else { 0.0 };
    let (fraction, message) = match p.stage {
        "load_vision" => (
            0.04 * p.current as f64 / p.total as f64,
            format!("首次加载视觉编码器 · {} / {}", p.current, p.total),
        ),
        "reference" => {
            track.reference_index = p.current;
            (0.04, format!("理解参考图片 · {} / {}", p.current, p.total))
        }
        "vision" | "vae_encode" => {
            let portion = if p.stage == "vision" { 0.0 } else { 0.5 };
            let fraction = 0.04
                + 0.21
                    * ((track.reference_index.saturating_sub(1)) as f64
                        + portion
                        + 0.5 * p.current as f64 / p.total as f64)
                    / options.images.len().max(1) as f64;
            (
                fraction,
                format!(
                    "{}参考图 {} · {} / {}",
                    if p.stage == "vision" {
                        "理解"
                    } else {
                        "编码"
                    },
                    track.reference_index,
                    p.current,
                    p.total
                ),
            )
        }
        "text" | "negative_text" => (
            base + 0.08
                * (if p.stage == "negative_text" { 1.0 } else { 0.0 }
                    + p.current as f64 / p.total as f64)
                / branches,
            format!(
                "理解{} · {} / {}",
                if p.stage == "negative_text" {
                    "负向提示词"
                } else {
                    "图片描述"
                },
                p.current,
                p.total
            ),
        ),
        "step" => {
            track.step = p.current;
            (
                base + 0.08 + (0.82 - base) * (track.step - 1) as f64 / total as f64,
                format!(
                    "{} · 第 {} / {total} 步",
                    if options.observation {
                        "去噪观察"
                    } else {
                        "生成图像"
                    },
                    track.step
                ),
            )
        }
        "denoise_layer" | "negative_denoise_layer" => (
            base + 0.08
                + (0.82 - base)
                    * ((track.step - 1) as f64
                        + (if p.stage == "negative_denoise_layer" {
                            1.0
                        } else {
                            0.0
                        } + p.current as f64 / p.total as f64)
                            / branches)
                    / total as f64,
            format!(
                "{} · 第 {} / {total} 步 · {}{} / {}",
                if options.observation {
                    "去噪观察"
                } else {
                    "生成图像"
                },
                track.step,
                if p.stage == "negative_denoise_layer" {
                    "负向 "
                } else {
                    ""
                },
                p.current,
                p.total
            ),
        ),
        _ if options.observation => (
            base + 0.08 + (0.92 - base) * track.step.max(1) as f64 / total.max(1) as f64,
            format!(
                "解码观察格 · 第 {} / {total} 步 · {} / {}",
                track.step.max(1),
                p.current,
                p.total
            ),
        ),
        _ => (
            0.90 + 0.10 * p.current as f64 / p.total as f64,
            format!("解码图像 · {} / {}", p.current, p.total),
        ),
    };
    let mut update = Update::status("progress", message);
    track.fraction_seen = track.fraction_seen.max(fraction);
    update.progress = track.fraction_seen;
    context.updates.publish(update);
}
