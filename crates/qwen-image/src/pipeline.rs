use crate::{
    Error, Precision, Progress, Result, check_cancel,
    config::Config,
    dimensions,
    dit::{Cache, Dit, ReferenceLatent},
    reference,
    text::TextEncoder,
    vae::Vae,
    vision::Vision,
};
use burn::tensor::{FloatDType, Tensor, TensorData, backend::Backend};
use std::{
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
    time::Instant,
};

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Options {
    pub prompt: String,
    /// Some("") requests the official unconditional branch; None disables CFG.
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default = "default_cfg_scale")]
    pub true_cfg_scale: f32,
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    #[serde(default)]
    pub images: Vec<PathBuf>,
    #[serde(default = "default_reference_resolution")]
    pub reference_resolution: usize,
    /// Desktop denoising observation. Ordinary generation leaves this off.
    #[serde(default)]
    pub observation: bool,
}
fn default_cfg_scale() -> f32 {
    1.0
}
fn default_reference_resolution() -> usize {
    1024
}
impl Default for Options {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: None,
            true_cfg_scale: default_cfg_scale(),
            width: dimensions::DEFAULT_SIDE,
            height: dimensions::DEFAULT_SIDE,
            seed: 42,
            images: Vec::new(),
            reference_resolution: default_reference_resolution(),
            observation: false,
        }
    }
}
impl Options {
    pub fn output_dimensions(&self) -> Result<(usize, usize)> {
        dimensions::resolve(self.width, self.height)
    }

    pub fn guidance_enabled(&self) -> bool {
        self.true_cfg_scale > 1.0 && self.negative_prompt.is_some()
    }

    pub fn validate(&self) -> Result<()> {
        if self.prompt.trim().is_empty() {
            return Err(Error::Config("请输入图片描述".into()));
        }
        if self.prompt.len() > 32768 || self.prompt.contains("<|") {
            return Err(Error::Config("描述过长或包含模型控制标记".into()));
        }
        if self
            .negative_prompt
            .as_ref()
            .is_some_and(|s| s.len() > 32768 || s.contains("<|"))
        {
            return Err(Error::Config("负向提示词过长或包含模型控制标记".into()));
        }
        // Product guardrail, not an architectural limit of the checkpoint.
        if !self.true_cfg_scale.is_finite() || !(1.0..=20.0).contains(&self.true_cfg_scale) {
            return Err(Error::Config(
                "引导强度须为 1–20 的有限数值；大于 1 时启用负向引导".into(),
            ));
        }
        if self.images.len() > reference::MAX_REFERENCES {
            return Err(Error::Config(format!(
                "最多支持 {} 张参考图",
                reference::MAX_REFERENCES
            )));
        }
        if !self.images.is_empty()
            && (!(256..=1024).contains(&self.reference_resolution)
                || self.reference_resolution % 32 != 0)
        {
            return Err(Error::Config(
                "参考图分辨率须为 256–1024 内 32 的倍数".into(),
            ));
        }
        self.output_dimensions()?;
        Ok(())
    }
}

#[derive(serde::Serialize)]
pub struct Report {
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    pub steps: usize,
    pub reference_count: usize,
    pub reference_resolution: usize,
    pub guidance_enabled: bool,
    pub true_cfg_scale: f32,
    pub reference_seconds: f64,
    pub text_seconds: f64,
    pub denoise_seconds: f64,
    pub decode_seconds: f64,
    pub elapsed_seconds: f64,
}
pub struct GeneratedImage {
    pub image: image::RgbaImage,
    pub report: Report,
    source_images: Vec<PathBuf>,
}
impl GeneratedImage {
    /// Keep the previous destination intact until the complete PNG is written.
    pub fn save_png(&self, path: &Path) -> Result<()> {
        validate_output_path(path, &self.source_images)?;
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension(format!("png.{}.part", std::process::id()));
        let result = (|| {
            self.image
                .save_with_format(&temporary, image::ImageFormat::Png)?;
            std::fs::rename(&temporary, path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(temporary);
        }
        result
    }
}

pub fn validate_png_path(path: &Path) -> Result<()> {
    if path
        .extension()
        .and_then(|s| s.to_str())
        .is_none_or(|s| !s.eq_ignore_ascii_case("png"))
    {
        return Err(Error::Config("请使用 .png 扩展名以保留透明通道".into()));
    }
    Ok(())
}

pub fn validate_output_path(path: &Path, sources: &[PathBuf]) -> Result<()> {
    validate_png_path(path)?;
    if let Ok(destination) = path.canonicalize() {
        if sources
            .iter()
            .any(|p| p.canonicalize().is_ok_and(|p| p == destination))
        {
            return Err(Error::Config("请另选保存位置，避免覆盖参考图片".into()));
        }
    }
    Ok(())
}

pub struct Model<B: Backend> {
    pub config: Config,
    pub text: TextEncoder<B>,
    pub dit: Dit<B>,
    pub vae: Vae<B>,
    vision: Option<Vision<B>>,
    minicpm_1b: Option<crate::minicpm_condition::MiniCpm1bText>,
    root: PathBuf,
    precision: Precision,
    device: B::Device,
}
impl<B: Backend> Model<B> {
    pub fn load(
        root: &Path,
        precision: Precision,
        device: &B::Device,
        cancel: &AtomicBool,
        mut progress: impl FnMut(Progress),
    ) -> Result<Self> {
        check_cancel(cancel)?;
        let config = Config::load(root)?;
        check_cancel(cancel)?;
        let text = TextEncoder::load(
            root,
            config.encoder.text_config.clone(),
            precision,
            device,
            cancel,
            &mut progress,
        )?;
        let dit = Dit::load(
            root,
            config.dit.clone(),
            precision,
            device,
            cancel,
            &mut progress,
        )?;
        let vae = Vae::load(root, &config.vae, precision, device, cancel, &mut progress)?;
        B::sync(device).map_err(|e| Error::Config(e.to_string()))?;
        check_cancel(cancel)?;
        Ok(Self {
            config,
            text,
            dit,
            vae,
            vision: None,
            minicpm_1b: None,
            root: root.to_owned(),
            precision,
            device: device.clone(),
        })
    }

    pub fn generate(
        &mut self,
        options: &Options,
        cancel: &AtomicBool,
        mut progress: impl FnMut(Progress),
    ) -> Result<GeneratedImage> {
        let kind = options.request_kind()?;
        if kind.decodes_after_each_step()
            || kind.loads_minicpm_v()
            || kind.reads_projection_weights()
            || kind.trajectories() != 1
        {
            return Err(Error::Config(
                "普通生成只加载官方文本塔，只跑一条扩散轨迹，并只在结束时解码一次".into(),
            ));
        }
        let mut state = self.prepare_generation(options, cancel, &mut progress, None)?;
        while state.step < self.config.sigmas.len() {
            check_cancel(cancel)?;
            progress(Progress {
                stage: "step",
                current: state.step + 1,
                total: self.config.sigmas.len(),
            });
            self.advance_generations(&mut [(&mut state, cancel)], &mut progress)?;
        }
        self.finish_generation(state, cancel, &mut progress)
    }

    /// Two trajectories: official text, then the fitted MiniCPM-V-4.7-1B projection.
    /// They share the prompt, seed, canvas and initial noise. Each keeps its own
    /// attention cache. Guidance stays off.
    pub fn observe(
        &mut self,
        options: &Options,
        cancel: &AtomicBool,
        mut progress: impl FnMut(Progress),
        mut on_cell: impl FnMut(crate::ObservationRow, usize, &image::RgbaImage),
    ) -> Result<DenoisingObservation> {
        let kind = options.request_kind()?;
        let refused = kind.negative_guidance()
            || !kind.loads_minicpm_v()
            || !kind.reads_projection_weights()
            || kind.trajectories() != 2
            || !kind.decodes_after_each_step();
        let RequestKind::OfficialObservation(request) = kind else {
            return Err(Error::Config("去噪观察尚未开启".into()));
        };
        if refused {
            return Err(Error::Config(
                "去噪观察要同时跑官方文本条件和 MiniCPM-V-4.7-1B 投影条件，且没有负向引导".into(),
            ));
        }
        if self.config.sigmas.len() != DEFAULT_TURBO_STEPS {
            return Err(Error::Config(format!(
                "去噪观察要求默认 Turbo 的 {DEFAULT_TURBO_STEPS} 步调度"
            )));
        }
        let mut observation = DenoisingObservation::open(
            request,
            self.config.sigmas.len(),
            crate::projection::ProjectionSource::Weights(
                &crate::projection::default_projection_weights_path(),
            ),
        )?;
        let projection = observation.projection().cloned().ok_or_else(|| {
            Error::Config(
                "去噪观察缺少 MiniCPM-V-4.7-1B 投影权重，不能用未训练投影出图，也不会退回 JSON 改写"
                    .into(),
            )
        })?;
        let noise = observation.trajectories()[0].initial_noise().to_vec();
        let last_layer = self.minicpm_1b()?.last_layer(options.prompt.as_str())?;
        let mut official =
            self.prepare_generation(options, cancel, &mut progress, Some(noise.clone()))?;
        if official.negative.is_some() {
            return Err(Error::Config("去噪观察没有负向条件分支".into()));
        }
        let official_length = official.text.dims()[1];
        if official.text.dims()[2] != crate::projection::CONDITION_WIDTH {
            return Err(Error::Config(
                "官方文本条件的宽度不是 4096，无法对齐 MiniCPM-V-4.7-1B 投影".into(),
            ));
        }
        let projected = projection.project_last_layer(&last_layer, official_length)?;
        let mut minicpm = self.trajectory_with_condition(&official, &noise, projected.values())?;
        while official.step < self.config.sigmas.len() {
            check_cancel(cancel)?;
            progress(Progress {
                stage: "step",
                current: official.step + 1,
                total: self.config.sigmas.len(),
            });
            self.advance_generations(
                &mut [(&mut official, cancel), (&mut minicpm, cancel)],
                &mut progress,
            )?;
            for (row, state) in [
                (crate::ObservationRow::Official, &official),
                (crate::ObservationRow::MiniCpm1b, &minicpm),
            ] {
                let image = self.decode_latents(
                    state.latents.clone(),
                    state.width,
                    state.height,
                    state.h,
                    state.w,
                    state.channels,
                    cancel,
                    &mut progress,
                )?;
                let step = state.step;
                observation.retain(row, step, image)?;
                let cell = &observation.cells(row)[step - 1];
                on_cell(row, cell.step, &cell.image);
            }
        }
        Ok(observation)
    }

    fn minicpm_1b(&mut self) -> Result<&crate::minicpm_condition::MiniCpm1bText> {
        if self.minicpm_1b.is_none() {
            let dir = crate::minicpm_condition::default_minicpm_1b_dir(&self.root);
            self.minicpm_1b = Some(crate::minicpm_condition::MiniCpm1bText::load(&dir)?);
        }
        Ok(self.minicpm_1b.as_ref().unwrap())
    }

    fn trajectory_with_condition(
        &self,
        official: &Generation<B>,
        noise: &[f32],
        condition: &[f32],
    ) -> Result<Generation<B>> {
        let length = official.text.dims()[1];
        if condition.len() != length * crate::projection::CONDITION_WIDTH {
            return Err(Error::Config(
                "MiniCPM-V-4.7-1B 投影与官方文本条件长度不一致".into(),
            ));
        }
        let text = Tensor::<B, 3>::from_data(
            TensorData::new(
                condition.to_vec(),
                [1, length, crate::projection::CONDITION_WIDTH],
            ),
            &self.device,
        )
        .cast(self.precision.dtype());
        let latents = Tensor::<B, 3>::from_data(
            TensorData::new(
                noise.to_vec(),
                [1, official.channels, official.h * official.w],
            ),
            &self.device,
        )
        .swap_dims(1, 2)
        .cast(self.precision.dtype());
        Ok(Generation {
            options: official.options.clone(),
            start: official.start,
            reference_seconds: official.reference_seconds,
            text_seconds: official.text_seconds,
            denoise_start: official.denoise_start,
            latents,
            text,
            references: Vec::new(),
            spans: Vec::new(),
            negative: None,
            cache: Cache::new(self.config.dit.num_layers),
            step: 0,
            width: official.width,
            height: official.height,
            h: official.h,
            w: official.w,
            channels: official.channels,
        })
    }

    fn prepare_generation(
        &mut self,
        options: &Options,
        cancel: &AtomicBool,
        mut progress: &mut impl FnMut(Progress),
        noise: Option<Vec<f32>>,
    ) -> Result<Generation<B>> {
        options.validate()?;
        let (width, height) = options.output_dimensions()?;
        check_cancel(cancel)?;
        let start = Instant::now();
        let mut references = Vec::new();
        let mut features = Vec::new();
        let mut grids = Vec::new();
        if !options.images.is_empty() {
            let prepared = options
                .images
                .iter()
                .map(|path| {
                    check_cancel(cancel)?;
                    reference::prepare(
                        &reference::open(path)?,
                        options.reference_resolution,
                        cancel,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            if self.vision.is_none() {
                self.vision = Some(Vision::load(
                    &self.root,
                    self.config.encoder.vision_config.clone(),
                    self.precision,
                    &self.device,
                    cancel,
                    &mut progress,
                )?);
            }
            let mean = Tensor::<B, 4>::from_data(
                TensorData::new(self.config.vae.latents_mean.clone(), [1, 64, 1, 1]),
                &self.device,
            )
            .cast(self.precision.dtype());
            let std = Tensor::<B, 4>::from_data(
                TensorData::new(self.config.vae.latents_std.clone(), [1, 64, 1, 1]),
                &self.device,
            )
            .cast(self.precision.dtype());
            for (i, image) in prepared.into_iter().enumerate() {
                check_cancel(cancel)?;
                progress(Progress {
                    stage: "reference",
                    current: i + 1,
                    total: options.images.len(),
                });
                let h = image.grid_height;
                let w = image.grid_width;
                let patches = Tensor::<B, 3>::from_data(
                    TensorData::new(image.patches, [1, h * w, 1536]),
                    &self.device,
                )
                .cast(self.precision.dtype());
                let feature = self.vision.as_ref().unwrap().encode(
                    patches,
                    h,
                    w,
                    cancel,
                    &mut progress,
                    &mut |_, _| Ok(()),
                )?;
                grids.push((feature.height, feature.width));
                features.push(feature);
                let rgba = Tensor::<B, 4>::from_data(
                    TensorData::new(image.vae_pixels, [1, 4, h * 16, w * 16]),
                    &self.device,
                )
                .cast(self.precision.dtype());
                let encoded = self.vae.encode(rgba, cancel, &mut progress)?;
                let latents = ((encoded - mean.clone()) / std.clone())
                    .reshape([1, 64, h * w])
                    .swap_dims(1, 2);
                references.push(ReferenceLatent {
                    latents,
                    height: h,
                    width: w,
                });
                B::sync(&self.device).map_err(|e| Error::Config(e.to_string()))?;
            }
        }
        let reference_seconds = start.elapsed().as_secs_f64();
        let text_start = Instant::now();
        let guided = options.guidance_enabled();
        let layout = self.text.layout(&options.prompt, &grids)?;
        let spans = layout.condition_spans();
        let text = self.text.encode_multimodal(
            &layout,
            &features,
            cancel,
            &mut progress,
            &mut |_, _| Ok(()),
        )?;
        // Reuse the same visual features and VAE latents. The two prompts must
        // have independent layouts and DiT KV caches, including image editing.
        let negative = if guided {
            check_cancel(cancel)?;
            let layout = self
                .text
                .layout(options.negative_prompt.as_deref().unwrap(), &grids)?;
            let spans = layout.condition_spans();
            let text = self.text.encode_multimodal(
                &layout,
                &features,
                cancel,
                &mut |mut p| {
                    p.stage = "negative_text";
                    progress(p);
                },
                &mut |_, _| Ok(()),
            )?;
            Some((text, spans, Cache::new(self.config.dit.num_layers)))
        } else {
            None
        };
        drop(features);
        B::sync(&self.device).map_err(|e| Error::Config(e.to_string()))?;
        let text_seconds = text_start.elapsed().as_secs_f64();
        let h = height / 16;
        let w = width / 16;
        let channels = self.config.vae.z_dim;
        // Channel-major noise. Observation passes this same vector back in so the
        // official trajectory and any later one start from identical samples.
        let noise = match noise {
            Some(noise) => noise,
            None => observation::latent_noise(channels, height, width, options.seed),
        };
        if noise.len() != channels * h * w {
            return Err(Error::Config("初始噪声长度与画布不一致".into()));
        }
        let latents =
            Tensor::<B, 3>::from_data(TensorData::new(noise, [1, channels, h * w]), &self.device)
                .swap_dims(1, 2)
                .cast(self.precision.dtype());
        let cache = Cache::new(self.config.dit.num_layers);

        let denoise_start = Instant::now();
        Ok(Generation {
            options: options.clone(),
            start,
            reference_seconds,
            text_seconds,
            denoise_start,
            latents,
            text,
            references,
            spans,
            negative,
            cache,
            step: 0,
            width,
            height,
            h,
            w,
            channels,
        })
    }
    fn finish_generation(
        &self,
        state: Generation<B>,
        cancel: &AtomicBool,
        progress: &mut impl FnMut(Progress),
    ) -> Result<GeneratedImage> {
        let Generation {
            options,
            start,
            reference_seconds,
            text_seconds,
            denoise_start,
            latents,
            cache,
            negative,
            width,
            height,
            h,
            w,
            channels,
            ..
        } = state;
        let guided = options.guidance_enabled();
        let denoise_seconds = denoise_start.elapsed().as_secs_f64();
        drop(cache);
        drop(negative);
        let decode_start = Instant::now();
        let image =
            self.decode_latents(latents, width, height, h, w, channels, cancel, progress)?;
        let decode_seconds = decode_start.elapsed().as_secs_f64();
        Ok(GeneratedImage {
            image,
            source_images: options.images.clone(),
            report: Report {
                width,
                height,
                seed: options.seed,
                steps: self.config.sigmas.len(),
                reference_count: options.images.len(),
                reference_resolution: options.reference_resolution,
                guidance_enabled: guided,
                true_cfg_scale: options.true_cfg_scale,
                reference_seconds,
                text_seconds,
                denoise_seconds,
                decode_seconds,
                elapsed_seconds: start.elapsed().as_secs_f64(),
            },
        })
    }

    fn decode_latents(
        &self,
        latents: Tensor<B, 3>,
        width: usize,
        height: usize,
        h: usize,
        w: usize,
        channels: usize,
        cancel: &AtomicBool,
        progress: &mut impl FnMut(Progress),
    ) -> Result<image::RgbaImage> {
        check_cancel(cancel)?;
        let latents = latents.swap_dims(1, 2).reshape([1, channels, h, w]);
        let mean = Tensor::<B, 4>::from_data(
            TensorData::new(self.config.vae.latents_mean.clone(), [1, channels, 1, 1]),
            &self.device,
        )
        .cast(self.precision.dtype());
        let std = Tensor::<B, 4>::from_data(
            TensorData::new(self.config.vae.latents_std.clone(), [1, channels, 1, 1]),
            &self.device,
        )
        .cast(self.precision.dtype());
        let rgba = self.vae.decode(latents * std + mean, cancel, progress)?;
        let shape = rgba.dims();
        if shape != [1, 4, height, width] {
            return Err(Error::Config(format!("VAE 输出尺寸异常：{shape:?}")));
        }
        let values = rgba
            .cast(FloatDType::F32)
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| Error::Config(format!("读取图像失败：{e:?}")))?;
        check_cancel(cancel)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Config("模型产生了非有限像素，未保存无效图像".into()));
        }
        let pixels = width * height;
        let mut image = image::RgbaImage::new(width as u32, height as u32);
        for (i, pixel) in image.pixels_mut().enumerate() {
            for channel in 0..4 {
                pixel[channel] = ((values[channel * pixels + i] + 1.0) * 127.5)
                    .clamp(0.0, 255.0)
                    .round_ties_even() as u8;
            }
        }
        Ok(image)
    }

    fn advance_generations(
        &self,
        work: &mut [(&mut Generation<B>, &AtomicBool)],
        progress: &mut impl FnMut(Progress),
    ) -> Result<()> {
        use crate::dit::Prediction;
        let mut predictions: Vec<_> = work
            .iter_mut()
            .map(|(state, cancel)| Prediction {
                latents: state.latents.clone(),
                text: state.text.clone(),
                references: &state.references,
                spans: &state.spans,
                sigma: self.config.sigmas[state.step],
                height: state.h,
                width: state.w,
                cache: &mut state.cache,
                cancel,
            })
            .collect();
        let conditional = self.dit.predict_many(&mut predictions, progress)?;
        drop(predictions);
        let mut indices = Vec::new();
        let mut negatives = Vec::new();
        for (i, (state, cancel)) in work.iter_mut().enumerate() {
            if cancel.load(std::sync::atomic::Ordering::Acquire) {
                continue;
            }
            if let Some((text, spans, cache)) = &mut state.negative {
                indices.push(i);
                negatives.push(Prediction {
                    latents: state.latents.clone(),
                    text: text.clone(),
                    references: &state.references,
                    spans,
                    sigma: self.config.sigmas[state.step],
                    height: state.h,
                    width: state.w,
                    cache,
                    cancel,
                });
            }
        }
        let uncond = if negatives.is_empty() {
            Vec::new()
        } else {
            match self.dit.predict_many(&mut negatives, &mut |mut event| {
                event.stage = "negative_denoise_layer";
                progress(event);
            }) {
                Ok(values) => values,
                Err(Error::Cancelled)
                    if negatives
                        .iter()
                        .all(|item| item.cancel.load(std::sync::atomic::Ordering::Acquire)) =>
                {
                    Vec::new()
                }
                Err(error) => return Err(error),
            }
        };
        drop(negatives);
        let mut uncond: std::collections::BTreeMap<_, _> =
            indices.into_iter().zip(uncond).collect();
        for (i, (state, cancel)) in work.iter_mut().enumerate() {
            if cancel.load(std::sync::atomic::Ordering::Acquire) {
                continue;
            }
            let noise = match uncond.remove(&i) {
                Some(u) => classifier_free_guidance(
                    conditional[i].clone(),
                    u,
                    state.options.true_cfg_scale,
                    self.precision,
                ),
                None => conditional[i].clone(),
            };
            let sigma = self.config.sigmas[state.step];
            let next = self
                .config
                .sigmas
                .get(state.step + 1)
                .copied()
                .unwrap_or(0.0);
            state.latents = euler_step(state.latents.clone(), noise, next - sigma, self.precision);
            state.step += 1;
        }
        B::sync(&self.device).map_err(|e| Error::Config(e.to_string()))?;
        Ok(())
    }
}

/// Match PyTorch's three FP16 arithmetic boundaries rather than silently
/// merging the expression into one full-precision multiply-add.
pub fn classifier_free_guidance<B: Backend>(
    positive: Tensor<B, 3>,
    negative: Tensor<B, 3>,
    scale: f32,
    precision: Precision,
) -> Tensor<B, 3> {
    let delta = (positive.cast(FloatDType::F32) - negative.clone().cast(FloatDType::F32))
        .cast(precision.dtype());
    let scaled = (delta.cast(FloatDType::F32) * scale).cast(precision.dtype());
    (negative.cast(FloatDType::F32) + scaled.cast(FloatDType::F32)).cast(precision.dtype())
}

pub fn euler_step<B: Backend>(
    latents: Tensor<B, 3>,
    velocity: Tensor<B, 3>,
    delta: f32,
    precision: Precision,
) -> Tensor<B, 3> {
    (latents.cast(FloatDType::F32) + velocity.cast(FloatDType::F32) * delta).cast(precision.dtype())
}

/// Stable per-request RNG; intentionally independent of PyTorch's generator.
pub fn gaussian_noise(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut uniform = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = state;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^= x >> 31;
        ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut output = Vec::with_capacity(count);
    while output.len() < count {
        let radius = (-2.0 * uniform().ln()).sqrt();
        let angle = std::f64::consts::TAU * uniform();
        output.push((radius * angle.cos()) as f32);
        if output.len() < count {
            output.push((radius * angle.sin()) as f32);
        }
    }
    output
}

struct Generation<B: Backend> {
    options: Options,
    start: Instant,
    reference_seconds: f64,
    text_seconds: f64,
    denoise_start: Instant,
    latents: Tensor<B, 3>,
    text: Tensor<B, 3>,
    references: Vec<ReferenceLatent<B>>,
    spans: Vec<crate::text::ImageSpan>,
    negative: Option<(Tensor<B, 3>, Vec<crate::text::ImageSpan>, Cache<B>)>,
    cache: Cache<B>,
    step: usize,
    width: usize,
    height: usize,
    h: usize,
    w: usize,
    channels: usize,
}
#[path = "batch.rs"]
mod batch;
pub use batch::{BatchStats, ImageBatch};
#[path = "observation.rs"]
mod observation;
pub use observation::{
    DEFAULT_TURBO_STEPS, DenoisingObservation, ObservationCell, ObservationRequest, ObservationRow,
    OfficialObservation, RequestKind, Trajectory,
};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn per_request_noise_is_repeatable_finite_and_independent_of_prior_requests() {
        let a = gaussian_noise(10001, 42);
        let _ = gaussian_noise(13, 99);
        assert_eq!(a, gaussian_noise(10001, 42));
        assert_ne!(a[..13], gaussian_noise(13, 43));
        assert!(a.iter().all(|v| v.is_finite()));
        let mean = a.iter().sum::<f32>() / a.len() as f32;
        let square = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32;
        assert!(mean.abs() < 0.04 && (square - 1.0).abs() < 0.05);
    }
    #[test]
    fn requests_resolve_output_dimensions_and_reject_invalid_inputs() {
        let mut o = Options {
            prompt: "一只猫".into(),
            ..Default::default()
        };
        assert!(o.validate().is_ok());
        o.width = 1025;
        assert!(o.validate().is_ok());
        assert_eq!(o.output_dimensions().unwrap(), (1024, 2048));
        o.width = 2753;
        assert!(o.validate().is_err());
        o.width = 1024;
        o.prompt = "<|image_pad|>".into();
        assert!(o.validate().is_err());
    }

    #[test]
    fn guidance_is_opt_in_and_empty_negative_is_a_valid_condition() {
        let mut o: Options =
            serde_json::from_str(r#"{"prompt":"茶壶","width":512,"height":512,"seed":42}"#)
                .unwrap();
        assert!(o.validate().is_ok());
        assert!(!o.guidance_enabled());
        o.true_cfg_scale = 2.0;
        assert!(!o.guidance_enabled());
        o.negative_prompt = Some(String::new());
        assert!(o.guidance_enabled() && o.validate().is_ok());
        o.true_cfg_scale = 1.0;
        assert!(!o.guidance_enabled());
        o.negative_prompt = Some("<|im_end|>".into());
        assert!(o.validate().is_err());
        o.negative_prompt = None;
        for value in [f32::NAN, f32::INFINITY, 0.0, -1.0, 20.1] {
            o.true_cfg_scale = value;
            assert!(o.validate().is_err());
        }
    }

    fn plain_observation() -> Options {
        Options {
            prompt: "一只茶壶".into(),
            width: 512,
            height: 512,
            seed: 7,
            observation: true,
            ..Default::default()
        }
    }

    #[test]
    fn observation_accepts_plain_text_with_guidance_off_and_refuses_the_other_inputs() {
        let request = ObservationRequest::accept(&plain_observation()).unwrap();
        assert_eq!(request.prompt(), "一只茶壶");
        assert_eq!(request.canvas(), (512, 512));
        assert_eq!(request.seed(), 7);
        assert!(!request.guidance_enabled());

        let mut with_reference = plain_observation();
        with_reference.images.push("ref.png".into());
        let error = ObservationRequest::accept(&with_reference)
            .unwrap_err()
            .to_string();
        assert!(error.contains("参考图"), "{error}");

        let mut with_negative = plain_observation();
        with_negative.negative_prompt = Some(String::new());
        let error = ObservationRequest::accept(&with_negative)
            .unwrap_err()
            .to_string();
        assert!(error.contains("负向"), "{error}");

        let mut with_guidance = plain_observation();
        with_guidance.true_cfg_scale = 2.0;
        let error = ObservationRequest::accept(&with_guidance)
            .unwrap_err()
            .to_string();
        assert!(error.contains("引导"), "{error}");

        // These inputs stay valid for an ordinary request.
        for mut ordinary in [with_reference, with_negative, with_guidance] {
            ordinary.observation = false;
            assert!(ordinary.validate().is_ok());
        }
        let mut guided = plain_observation();
        guided.observation = false;
        guided.negative_prompt = Some("模糊".into());
        guided.true_cfg_scale = 2.0;
        assert!(guided.guidance_enabled());
        assert!(guided.validate().is_ok());
    }

    #[test]
    fn official_observation_retains_one_canvas_decode_per_default_turbo_step() {
        assert_eq!(DEFAULT_TURBO_STEPS, 8);
        let request = ObservationRequest::accept(&plain_observation()).unwrap();
        let mut row = OfficialObservation::begin(request, DEFAULT_TURBO_STEPS).unwrap();
        assert_eq!(row.trajectories(), 1);
        assert!(!row.uses_negative_guidance());
        assert!(row.is_official_text());
        let expected = gaussian_noise(64 * (512 / 16) * (512 / 16), 7);
        assert_eq!(row.initial_noise(), expected.as_slice());
        assert!(
            OfficialObservation::begin(
                ObservationRequest::accept(&plain_observation()).unwrap(),
                7,
            )
            .is_err()
        );

        for step in 1..=8 {
            let image =
                image::RgbaImage::from_pixel(512, 512, image::Rgba([step as u8, 9, 8, 255]));
            row.retain(step, image).unwrap();
            assert_eq!(row.cells().len(), step);
            assert_eq!(row.cells()[0].image.get_pixel(0, 0).0, [1, 9, 8, 255]);
        }
        assert_eq!(
            row.cells().iter().map(|cell| cell.step).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6, 7, 8]
        );
        for cell in row.cells() {
            assert_eq!(cell.image.dimensions(), (512, 512));
        }
        let replacement = image::RgbaImage::from_pixel(512, 512, image::Rgba([99, 0, 0, 255]));
        assert!(row.retain(1, replacement).is_err());
        assert_eq!(row.cells().len(), 8);
        assert_eq!(row.cells()[0].image.get_pixel(0, 0).0[0], 1);

        let shrunk = image::RgbaImage::from_pixel(256, 256, image::Rgba([0, 0, 0, 255]));
        assert!(row.retain(9, shrunk).is_err());
        assert_eq!(row.cells()[7].image.dimensions(), (512, 512));

        let other = OfficialObservation::begin(
            ObservationRequest::accept(&Options {
                seed: 8,
                ..plain_observation()
            })
            .unwrap(),
            8,
        )
        .unwrap();
        assert_ne!(row.initial_noise(), other.initial_noise());
    }

    #[test]
    fn ordinary_generation_keeps_one_trajectory_and_one_final_decode() {
        let mut options = Options {
            prompt: "一只茶壶".into(),
            width: 512,
            height: 320,
            seed: 3,
            negative_prompt: Some("模糊".into()),
            true_cfg_scale: 2.0,
            images: vec!["ref.png".into()],
            observation: false,
            ..Default::default()
        };
        let kind = options.request_kind().unwrap();
        assert_eq!(kind.trajectories(), 1);
        assert_eq!(kind.retained_decodes(), 1);
        assert!(kind.negative_guidance());
        assert!(!kind.loads_minicpm_v());
        assert!(!kind.reads_projection_weights());
        assert!(!kind.decodes_after_each_step());

        options.observation = true;
        assert!(options.request_kind().is_err());

        options.images.clear();
        options.negative_prompt = None;
        options.true_cfg_scale = 1.0;
        options.observation = false;
        let kind = options.request_kind().unwrap();
        assert_eq!(kind.trajectories(), 1);
        assert!(!kind.loads_minicpm_v());
        assert!(!kind.reads_projection_weights());
        assert!(!kind.decodes_after_each_step());
    }

    #[test]
    fn observation_adds_a_minicpm_1b_trajectory_on_the_official_noise() {
        let options = plain_observation();
        let kind = options.request_kind().unwrap();
        assert_eq!(kind.trajectories(), 2);
        assert_eq!(kind.retained_decodes(), 16);
        assert!(!kind.negative_guidance());
        assert!(kind.loads_minicpm_v());
        assert!(kind.reads_projection_weights());
        assert!(kind.decodes_after_each_step());

        let observation = DenoisingObservation::begin(
            ObservationRequest::accept(&options).unwrap(),
            DEFAULT_TURBO_STEPS,
        )
        .unwrap();
        let trajectories = observation.trajectories();
        assert_eq!(trajectories.len(), 2);
        let official = &trajectories[0];
        let minicpm = &trajectories[1];
        assert!(official.is_official_text());
        assert!(minicpm.is_minicpm_v_4_7_1b());
        assert_eq!(official.prompt(), "一只茶壶");
        assert_eq!(official.prompt(), minicpm.prompt());
        assert_eq!(official.seed(), minicpm.seed());
        assert_eq!(official.seed(), 7);
        assert_eq!(official.canvas(), minicpm.canvas());
        assert_eq!(official.canvas(), (512, 512));
        assert!(!official.guidance_enabled());
        assert!(!minicpm.guidance_enabled());
        assert!(!observation.uses_negative_guidance());
        let expected = gaussian_noise(64 * (512 / 16) * (512 / 16), 7);
        assert_eq!(official.initial_noise(), expected.as_slice());
        assert_eq!(official.initial_noise(), minicpm.initial_noise());
        assert_ne!(official.attention_history(), minicpm.attention_history());

        let other = DenoisingObservation::begin(
            ObservationRequest::accept(&Options {
                seed: 8,
                ..plain_observation()
            })
            .unwrap(),
            DEFAULT_TURBO_STEPS,
        )
        .unwrap();
        assert_ne!(
            observation.trajectories()[0].initial_noise(),
            other.trajectories()[0].initial_noise()
        );
        assert_eq!(
            other.trajectories()[0].initial_noise(),
            other.trajectories()[1].initial_noise()
        );
    }

    #[test]
    fn minicpm_1b_projection_matches_the_official_condition_shape() {
        use crate::projection::{CONDITION_WIDTH, ConditionProjection, Sequence};

        let mut weight = vec![0.0; CONDITION_WIDTH];
        weight[0] = 2.0;
        let mut bias = vec![0.0; CONDITION_WIDTH];
        bias[0] = 1.0;
        let projection = ConditionProjection::from_linear(&weight, &bias).unwrap();

        let aligned = Sequence::last_layer(2, 1, vec![3.0, 4.0]).unwrap();
        let same_length = projection.project_last_layer(&aligned, 2).unwrap();
        assert_eq!(same_length.length(), 2);
        assert_eq!(same_length.width(), CONDITION_WIDTH);
        assert_eq!(same_length.get(0, 0), 7.0);
        assert_eq!(same_length.get(1, 0), 9.0);
        assert_eq!(same_length.get(0, 1), 0.0);

        let stretched = Sequence::last_layer(2, 1, vec![0.0, 10.0]).unwrap();
        let output = projection.project_last_layer(&stretched, 3).unwrap();
        assert_eq!(output.length(), 3);
        assert_eq!(output.width(), CONDITION_WIDTH);
        assert_eq!(output.get(0, 0), 1.0);
        assert_eq!(output.get(1, 0), 11.0);
        assert_eq!(output.get(2, 0), 21.0);
        assert_eq!(output.get(2, 4095), 0.0);

        let image = projection
            .project_image(&[9, 9, 9], 3)
            .unwrap_err()
            .to_string();
        assert!(image.contains("图像"), "{image}");
        let rewritten = projection
            .project_json_rewrite("{\"rewritten_prompt\":\"一只茶壶\"}", 3)
            .unwrap_err()
            .to_string();
        assert!(rewritten.contains("改写"), "{rewritten}");
    }

    #[test]
    fn observation_fails_explicitly_without_fitted_1b_projection_weights() {
        use crate::projection::{
            ConditionProjection, ProjectionSource, Sequence, text_to_image_split,
        };
        use std::time::{SystemTime, UNIX_EPOCH};

        let request = || ObservationRequest::accept(&plain_observation()).unwrap();
        let missing = std::env::temp_dir().join(format!(
            "omni-1b-missing-{}-{}.bin",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_file(&missing);
        let absent = match DenoisingObservation::open(
            request(),
            DEFAULT_TURBO_STEPS,
            ProjectionSource::Weights(&missing),
        ) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("missing weights must not open an observation"),
        };
        assert!(absent.contains("投影"), "{absent}");
        assert!(absent.contains("未训练"), "{absent}");
        assert!(absent.contains("改写"), "{absent}");
        assert!(!missing.exists());

        let untrained = match DenoisingObservation::open(
            request(),
            DEFAULT_TURBO_STEPS,
            ProjectionSource::Untrained,
        ) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("an untrained projection must not open an observation"),
        };
        assert!(untrained.contains("未训练"), "{untrained}");

        let rewrite = match DenoisingObservation::open(
            request(),
            DEFAULT_TURBO_STEPS,
            ProjectionSource::JsonRewrite,
        ) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("JSON rewrite must not open an observation"),
        };
        assert!(rewrite.contains("改写"), "{rewrite}");

        let init = ConditionProjection::deterministic_init(4).unwrap();
        let init_path = missing.with_extension("init.bin");
        assert!(init.save(&init_path).is_err());

        let (train_prompts, _) = text_to_image_split();
        let train: Vec<_> = (0..train_prompts.len())
            .map(|index| {
                let input = Sequence::last_layer(1, 4, vec![index as f32; 4]).unwrap();
                let target = Sequence::last_layer(1, 4096, vec![0.2; 4096]).unwrap();
                (input, target)
            })
            .collect();
        let fitted = ConditionProjection::fit(&train).unwrap();
        let fitted_path = missing.with_extension("fitted.bin");
        fitted.save(&fitted_path).unwrap();
        let opened = match DenoisingObservation::open(
            request(),
            DEFAULT_TURBO_STEPS,
            ProjectionSource::Weights(&fitted_path),
        ) {
            Ok(observation) => observation,
            Err(error) => panic!("fitted weights should open, got {error}"),
        };
        assert!(opened.cells(ObservationRow::Official).is_empty());
        assert!(opened.cells(ObservationRow::MiniCpm1b).is_empty());
        assert_eq!(opened.trajectories().len(), 2);
        let _ = std::fs::remove_file(&fitted_path);
    }
}
